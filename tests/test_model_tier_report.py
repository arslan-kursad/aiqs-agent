"""Unit tests for aiqs.model_tier_report — the cross-tier comparison tool.

Uses MOCKED two-arm runs (via run_two_arm(mock=True, model=..., provider=...)) purely to
synthesize two cheap on-disk "variants" and validate the REPORTING/aggregation machinery
(table building, per-model reconstruction via eval.crop_eval.evaluate_two_arm, wall-clock
lookup). This tests plumbing, not real model behaviour — no API calls anywhere.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aiqs.config import CropConfig
from aiqs.model_tier_report import build_report, format_table, format_detail_sections
from aiqs.vlm_crop import _ckpt_path, run_two_arm, write_results


@pytest.fixture()
def synthetic_run(tmp_path):
    """Mirrors tests/test_crop_experiment.py's fixture (images + peaked maps + scores)."""
    rng = np.random.default_rng(7)
    n = 40
    labels = np.array([0, 1] * (n // 2))
    scores = np.where(labels == 0, rng.normal(0.42, 0.10, n), rng.normal(0.58, 0.10, n))
    run_dir = tmp_path / "runs" / "synth_run"
    (run_dir / "anomaly_maps").mkdir(parents=True)
    img_dir = tmp_path / "datasets" / "synth" / "test" / "mix"
    img_dir.mkdir(parents=True)

    rows, manifest = [], []
    for i in range(n):
        img = img_dir / f"{i:03d}.png"
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img)
        m = np.zeros((16, 16))
        m[4:6, 10:12] = 5.0
        np.save(run_dir / "anomaly_maps" / f"mix__{i:03d}.npy", m)
        rows.append((str(img), int(labels[i]), float(scores[i])))
        manifest.append((str(img), f"anomaly_maps/mix__{i:03d}.npy"))

    pd.DataFrame(rows, columns=["image_path", "label", "score"]).to_csv(
        run_dir / "image_scores.csv", index=False)
    pd.DataFrame(manifest, columns=["image_path", "map_path"]).to_csv(
        run_dir / "anomaly_maps" / "manifest.csv", index=False)
    return tmp_path, run_dir


def _make_variant(synthetic_run, model, provider):
    repo_root, run_dir = synthetic_run
    _, _, states_a, states_b = run_two_arm(
        run_dir, k=2, mock=True, seed=42, folds=5, token_cost=0.0, lambda_grid=[0.0],
        repo_root=repo_root, max_items=None, crop_cfg=CropConfig(), model=model,
        provider=provider)
    return write_results(run_dir, states_a, states_b, mock=True, model=model,
                         provider=provider)


def test_build_report_finds_and_compares_two_variants(synthetic_run):
    _, run_dir = synthetic_run
    _make_variant(synthetic_run, "claude-haiku-4-5", "anthropic")
    _make_variant(synthetic_run, "gemini-x", "openai_compatible")

    rows = build_report(run_dir, include_mock=True)   # mock_ prefix here is just test setup
    assert len(rows) == 2
    models = {r["model"] for r in rows}
    assert models == {"claude-haiku-4-5", "gemini-x"}
    for r in rows:
        assert r["result"].arm_a.k_runs == 2
        # mock-sourced variants have no matching non-mock checkpoint -> wall-clock is n/a,
        # not a crash.
        assert r["wall_clock_min"] is None


def test_build_report_excludes_mock_by_default(synthetic_run):
    _, run_dir = synthetic_run
    _make_variant(synthetic_run, "claude-haiku-4-5", "anthropic")
    assert build_report(run_dir, include_mock=False) == []


def test_build_report_empty_run_dir_returns_empty_list(tmp_path):
    empty = tmp_path / "runs" / "nothing_here"
    empty.mkdir(parents=True)
    assert build_report(empty) == []


def test_format_table_and_detail_sections_render_without_crashing(synthetic_run):
    _, run_dir = synthetic_run
    _make_variant(synthetic_run, "claude-haiku-4-5", "anthropic")
    rows = build_report(run_dir, include_mock=True)
    table = format_table(rows)
    assert "claude-haiku-4-5" in table
    assert "model" in table and "provider" in table
    detail = format_detail_sections(rows)
    assert "claude-haiku-4-5" in detail
    assert "Verdict distribution" in detail


def test_wall_clock_minutes_reads_real_checkpoint_timestamps(synthetic_run):
    from aiqs.model_tier_report import _wall_clock_minutes

    _, run_dir = synthetic_run
    ckpt = _ckpt_path(run_dir, mock=False, model="claude-haiku-4-5", provider="anthropic")
    now = time.time()
    with open(ckpt, "w") as f:
        f.write(json.dumps({"arm": "A", "run": 0, "idx": 0, "timestamp": now}) + "\n")
        f.write(json.dumps({"arm": "A", "run": 0, "idx": 1, "timestamp": now + 90}) + "\n")
    minutes = _wall_clock_minutes(run_dir, "claude-haiku-4-5", "anthropic")
    assert minutes == pytest.approx(1.5, abs=1e-6)


def test_wall_clock_minutes_missing_checkpoint_is_none(synthetic_run):
    from aiqs.model_tier_report import _wall_clock_minutes

    _, run_dir = synthetic_run
    assert _wall_clock_minutes(run_dir, "no-such-model", "anthropic") is None
