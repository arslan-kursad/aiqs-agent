"""Regression tests for evaluate._load_weights (PatchCore checkpoint loading).

Guards the bug where a non-tensor entry in the checkpoint state_dict (PatchCore stores
strings/scalars/metadata alongside its memory-bank buffers) crashed the shape pass, and
where strict=False could silently swallow a missing learned buffer (degenerate ~0.5
AUROC). No anomalib model / GPU needed: a tiny nn.Module with a registered buffer + a
hand-built state_dict reproduces both conditions.
"""

from __future__ import annotations

import torch
from torch import nn

from aiqs.evaluate import _load_weights


class _Tiny(nn.Module):
    """A registered (initially empty, PatchCore-style) buffer + a learned parameter."""

    def __init__(self):
        super().__init__()
        self.register_buffer("memory_bank", torch.zeros(0))   # empty until 'fit'
        self.weight = nn.Parameter(torch.zeros(3))


def _save(tmp_path, state_dict):
    path = tmp_path / "ckpt.ckpt"
    torch.save({"state_dict": state_dict}, str(path))
    return path


def test_non_tensor_entries_do_not_crash_and_tensors_load(tmp_path):
    membank = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    weight = torch.tensor([1.0, 2.0, 3.0])
    ckpt = _save(tmp_path, {
        "memory_bank": membank,
        "weight": weight,
        "meta_backbone": "wide_resnet50_2",   # non-tensor string (the original crasher)
        "meta_num_neighbors": 9,              # non-tensor scalar
    })
    model = _Tiny()

    result = _load_weights(model, ckpt)   # must NOT raise on the non-tensor entries

    torch.testing.assert_close(model.memory_bank, membank)   # resized + restored
    torch.testing.assert_close(model.weight.detach(), weight)
    # The non-tensor metadata keys are simply ignored under strict=False.
    assert "memory_bank" not in result.missing_keys
    assert set(result.unexpected_keys) >= {"meta_backbone", "meta_num_neighbors"}


def test_present_buffer_loads_absent_buffer_reported_and_warns(tmp_path, recwarn):
    # Checkpoint OMITS memory_bank -> it must surface in missing_keys AND warn (the
    # silent-degeneration guard), not be swallowed.
    ckpt = _save(tmp_path, {"weight": torch.tensor([4.0, 5.0, 6.0])})
    model = _Tiny()

    result = _load_weights(model, ckpt)

    assert "memory_bank" in result.missing_keys
    assert any(issubclass(w.category, RuntimeWarning)
               and "memory_bank" in str(w.message) for w in recwarn)
