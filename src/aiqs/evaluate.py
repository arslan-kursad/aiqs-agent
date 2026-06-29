"""Evaluate a trained anomaly detector checkpoint and persist a baseline.

    uv run aiqs-eval --category screw

Computes detection metrics (image AUROC, pixel AUPRO, AUPIMO when it computes
cleanly), persists them + per-image scores to results/, and prints a summary.
The per-image scores are the bridge to the Phase-1 adjudication layer.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback, seed_everything

from aiqs._anomalib_compat import anomalib_major
from aiqs.config import add_common_args, config_from_args
from aiqs.data import build_datamodule
from aiqs.detector import build_eval_engine, build_model, checkpoint_path
from aiqs.eval import EvalResult, persist, print_baseline_summary
from aiqs.eval.metrics import canonicalize, metric_config


def _to_list(value):
    """Coerce a tensor / array / scalar / sequence to a flat python list."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return list(value)


def _field(data, names):
    """Fetch the first present field by dict-key or attribute, across aliases."""
    for n in names:
        if isinstance(data, dict) and n in data:
            return data[n]
        if hasattr(data, n):
            return getattr(data, n)
    return None


class ScoreCollector(Callback):
    """Capture per-image (path, gt-label, anomaly-score) during test()."""

    def __init__(self) -> None:
        self.paths: list = []
        self.labels: list = []
        self.scores: list = []

    def on_test_start(self, trainer, pl_module) -> None:  # reset per test() call
        self.paths, self.labels, self.scores = [], [], []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloader_idx: int = 0) -> None:
        data = outputs if outputs is not None else batch
        scores = _to_list(_field(data, ["pred_scores", "pred_score"]))
        labels = _to_list(_field(data, ["label", "gt_label", "labels"]))
        paths = _field(data, ["image_path", "image_paths"])
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if scores is not None:
            self.scores.extend(scores)
        if labels is not None:
            self.labels.extend(int(x) for x in labels)
        if paths is not None:
            self.paths.extend([str(p) for p in paths])

    def as_kwargs(self) -> dict:
        n = len(self.scores)
        if n == 0:
            return {}
        return {
            "image_scores": self.scores,
            "image_labels": self.labels if len(self.labels) == n else None,
            "image_paths": self.paths if len(self.paths) == n else None,
        }


def _load_weights(model, ckpt):
    """Load trained weights with strict=False. Returns the load result (missing_keys /
    unexpected_keys) so callers/tests can inspect what was and was not restored.

    The training checkpoint has no metric buffers (training ran with no metrics),
    but the eval model registers them (AUPRO/AUPIMO). A strict load via
    `engine.test(ckpt_path=...)` therefore fails on the missing metric keys. The
    metric buffers are accumulators, not learned weights, so loading the real
    weights non-strictly (and letting the metrics initialise fresh) is correct.

    PatchCore adds two wrinkles, both handled detector-agnostically (no model-name
    special-casing):
      * Its coreset `memory_bank` buffer is EMPTY in a freshly built model and only
        gets its shape during fit. `load_state_dict` errors on that size mismatch even
        with strict=False, so we resize matching buffers to the checkpoint shape before
        copying. (EfficientAD has no such buffer — a no-op there.)
      * Its checkpoint state_dict holds NON-TENSOR entries (strings / scalars /
        metadata). The shape pass therefore inspects only real tensors; load_state_dict
        ignores the rest under strict=False.

    strict=False can silently swallow a missing learned buffer (e.g. an unrestored
    `memory_bank` -> degenerate ~0.5 AUROC), so we WARN when any of the model's own
    registered buffers land in missing_keys instead of passing in silence.
    """
    state = torch.load(str(ckpt), map_location="cpu")
    state_dict = state.get("state_dict", state)
    # Only real tensors carry a meaningful .shape; PatchCore stores non-tensor metadata.
    ckpt_shapes = {k: v.shape for k, v in state_dict.items()
                   if isinstance(v, torch.Tensor)}
    for name, buf in model.named_buffers():
        if name in ckpt_shapes and buf.shape != ckpt_shapes[name]:
            buf.resize_(ckpt_shapes[name])
    result = model.load_state_dict(state_dict, strict=False)

    missing_buffers = [n for n, _ in model.named_buffers() if n in set(result.missing_keys)]
    if missing_buffers:
        warnings.warn(
            "load_state_dict (strict=False) did NOT restore these registered buffers "
            f"from the checkpoint: {missing_buffers}. A missing learned buffer (e.g. a "
            "PatchCore memory_bank) yields a degenerate detector — verify the checkpoint.",
            RuntimeWarning, stacklevel=2)
    if result.missing_keys or result.unexpected_keys:
        print(f"  [load] missing_keys={len(result.missing_keys)} "
              f"unexpected_keys={len(result.unexpected_keys)} (strict=False)")
    return result


def _run_test(cfg, datamodule, ckpt, collector, use_aupimo):
    image_metrics, pixel_metrics = metric_config(use_aupimo)
    engine = build_eval_engine(cfg, image_metrics, pixel_metrics,
                               callbacks=[collector])
    model = build_model(cfg)
    _load_weights(model, ckpt)
    return engine.test(model=model, datamodule=datamodule, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed, workers=True)

    ckpt = checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Train first: "
            f"`uv run aiqs-train --category {cfg.category}`."
        )

    extras: dict = {}
    maps = None

    if anomalib_major() >= 2:
        # anomalib 2.x (GPU host): metrics via Evaluator + ImageBatch predict, plus the
        # Phase-2B anomaly-map export. GPU-host-verified — see _detector_v2.py.
        from aiqs import _detector_v2

        metrics, score_kwargs, maps = _detector_v2.run_eval_export(cfg, ckpt)
    else:
        datamodule = build_datamodule(cfg)
        collector = ScoreCollector()
        # AUPIMO is available in anomalib 1.2.0 but treated as best-effort: if it does
        # not compute cleanly we fall back to AUROC/AUPRO and log it (per project rule).
        try:
            raw = _run_test(cfg, datamodule, ckpt, collector, use_aupimo=True)
        except Exception as exc:  # noqa: BLE001 - want any failure to trigger fallback
            extras["aupimo"] = (
                f"disabled — did not compute cleanly ({type(exc).__name__}: {exc}); "
                f"logged as a Phase-1 follow-up"
            )
            raw = _run_test(cfg, datamodule, ckpt, collector, use_aupimo=False)
        metrics = canonicalize(raw)
        score_kwargs = collector.as_kwargs()

    result = EvalResult(
        run_id=cfg.run_id,
        meta=cfg.as_flat_dict(),
        metrics=metrics,
        extras=extras,
        **score_kwargs,
    )

    config_yaml = Path(args.config).read_text() if Path(args.config).exists() else None
    run_dir = persist(result, cfg.output.results_dir, config_yaml=config_yaml)
    if maps:
        _write_anomaly_maps(run_dir, maps)
    print_baseline_summary(result, run_dir)


def _write_anomaly_maps(run_dir, maps: dict) -> None:
    """Persist per-image anomaly maps (.npy) for the Phase-2B crop instrument.

    Heavy per-pixel dumps are gitignored (`results/runs/*/anomaly_maps/`); a small
    manifest (image_path -> map file) is written so the crop step can pair each map with
    its original image. The parent-folder name disambiguates same-stem files across
    defect folders (e.g. several `000.png`).
    """
    maps_dir = Path(run_dir) / "anomaly_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_path, arr in maps.items():
        p = Path(image_path)
        name = f"{p.parent.name}__{p.stem}.npy"
        np.save(maps_dir / name, np.asarray(arr, dtype=np.float32))
        manifest.append((image_path, f"anomaly_maps/{name}"))
    with open(maps_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "map_path"])
        writer.writerows(manifest)
    print(f"  [maps] wrote {len(manifest)} anomaly maps -> {maps_dir}")


if __name__ == "__main__":
    main()
