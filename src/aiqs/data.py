"""MVTec datamodule construction — version-dispatched.

Public seam (``build_datamodule``) is stable across both anomalib stacks:

  * **anomalib 1.2** (pinned local stack): the ``MVTec`` datamodule, original MVTec AD
    only — the v1 body below. Exercised by `make smoke`/`make test`.
  * **anomalib 2.x** (GPU-host ``ad2`` extra): ``MVTecAD`` (original) or ``MVTecAD2``
    (the reality-gap MVTec AD 2), in ``_data_v2.py``. Imported ONLY when anomalib 2.x is
    installed; GPU-host-verified (the local 1.2 stack cannot install it).

No anomalib import at module top: that would crash on import under 2.x (the 1.2 ``MVTec``
class was removed). Each path imports lazily.
"""

from __future__ import annotations

from aiqs._anomalib_compat import anomalib_major
from aiqs.config import Config


def build_datamodule(cfg: Config):
    """Build the datamodule for the configured dataset/category."""
    if anomalib_major() >= 2:
        from aiqs import _data_v2

        return _data_v2.build_datamodule(cfg)
    return _build_datamodule_v1(cfg)


def _build_datamodule_v1(cfg: Config):
    """anomalib 1.2: original MVTec AD only (task=SEGMENTATION keeps GT masks for AUPRO/AUPIMO)."""
    from anomalib import TaskType
    from anomalib.data import MVTec

    if cfg.dataset.name != "mvtec":
        raise ValueError(
            f"The pinned anomalib 1.2 stack supports the 'mvtec' dataset only, got "
            f"'{cfg.dataset.name}'. MVTec AD 2 needs anomalib 2.x — install the GPU-host "
            f"`ad2` extra (`uv sync --extra ad2`) on a CUDA host."
        )
    return MVTec(
        root=cfg.dataset.root,
        category=cfg.category,
        train_batch_size=cfg.dataset.train_batch_size,
        eval_batch_size=cfg.dataset.eval_batch_size,
        num_workers=cfg.dataset.num_workers,
        task=TaskType.SEGMENTATION,
        image_size=tuple(cfg.dataset.image_size),
        seed=cfg.seed,
    )
