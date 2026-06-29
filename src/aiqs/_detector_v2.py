"""anomalib 2.x detector backend — model, train Engine, and the ANOMALY-MAP EXPORT.

GPU-HOST-VERIFIED, NOT RUN ON THE LOCAL 1.2 STACK
-------------------------------------------------
Imported only when ``anomalib.__version__`` >= 2 (a CUDA host with the ``ad2`` extra);
the pinned Intel-mac stack dispatches to the 1.2 bodies in ``detector.py`` instead. This
module encodes the verified 2.x API but is validated on the GPU host — treat every call
here as designed-not-locally-tested.

Verified 2.x deltas vs the 1.2 path (see CLAUDE.md Phase-2B Stage 1):
  * Engine no longer takes ``task=`` / ``image_metrics=`` / ``pixel_metrics=`` — metrics
    moved to an ``Evaluator``; here we read image-level AUROC/F1 straight off the
    collected predictions with sklearn (decision layer is image-level, detector-free).
  * ``engine.predict()`` returns ``ImageBatch`` objects: ``.pred_score`` / ``.pred_label``
    / ``.gt_label`` / ``.image_path`` / ``.anomaly_map`` (the per-pixel heatmap the crop
    instrument consumes).

The export is the CORE Stage-1 product: it writes the same image-level scores the Phase-1
decision layer consumes PLUS the per-image anomaly maps that ``crop.compute_crop`` turns
into VLM crops downstream. Maps are heavy and gitignored; only image scores are committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiqs.config import Config


def build_model(cfg: Config):
    from anomalib.models import EfficientAd, Patchcore

    name = cfg.model.name
    if name == "efficient_ad":
        return EfficientAd(imagenet_dir=cfg.model.imagenet_dir, model_size=cfg.model.size)
    if name == "patchcore":
        return Patchcore(
            backbone=cfg.model.backbone,
            layers=cfg.model.layers,
            coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
            num_neighbors=cfg.model.num_neighbors,
        )
    raise ValueError(f"Unsupported model '{name}'. Supported: 'efficient_ad', 'patchcore'.")


def build_train_engine(cfg: Config, root_dir: str, step_driven: bool):
    from anomalib.engine import Engine  # 2.x Engine: no task= argument

    common = dict(
        default_root_dir=root_dir,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=False,
    )
    if step_driven:
        return Engine(max_steps=cfg.training.max_steps, max_epochs=-1, **common)
    return Engine(max_steps=-1, max_epochs=1, **common)


def run_eval_export(cfg: Config, ckpt: Path):
    """Predict on the eval split; return (metrics, score_kwargs, maps).

    ``metrics``      — canonical dict (image_auroc / image_f1score; pixel_* left None,
                       a GPU-host ``Evaluator`` add-on is a TODO — see note below).
    ``score_kwargs`` — image_scores / image_labels / image_paths for EvalResult/persist.
    ``maps``         — {image_path: 2-D anomaly-map ndarray} for the crop instrument.
    """
    from anomalib.engine import Engine

    from aiqs.data import build_datamodule

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    engine = Engine(
        default_root_dir=str(Path(cfg.output.models_dir) / cfg.run_id),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=False,
    )
    batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(ckpt))

    scores, labels, paths, maps = [], [], [], {}
    for batch in batches or []:
        b_scores = _to_np(getattr(batch, "pred_score", None))
        b_labels = _to_np(getattr(batch, "gt_label", None))
        b_paths = _as_list(getattr(batch, "image_path", None))
        b_maps = getattr(batch, "anomaly_map", None)
        for i, path in enumerate(b_paths):
            scores.append(float(b_scores[i]))
            labels.append(int(b_labels[i]) if b_labels is not None else None)
            paths.append(str(path))
            if b_maps is not None:
                maps[str(path)] = np.asarray(_to_np(b_maps[i])).squeeze()

    metrics = _image_metrics(scores, labels)
    score_kwargs = {
        "image_scores": scores,
        "image_labels": labels if all(v is not None for v in labels) else None,
        "image_paths": paths,
    }
    # NOTE: pixel AUPRO/AUPIMO need the 2.x Evaluator (or per-pixel masks) — added on the
    # GPU host; the Stage-1/2 lever (substrate + decision) is image-level only.
    return metrics, score_kwargs, maps


def smoke(cfg) -> str:
    """Cheap 2.x API SHAKE-OUT before any full run: 1 train batch + 1 predict batch.

    The point (per the GPU-round guard in CLAUDE.md): fail in SECONDS on an API mismatch,
    not 40 minutes into a real train. Exercises the whole path that the real run uses —
    ``build_datamodule`` (incl. the AD2 auto-download), ``build_model``, ``engine.fit``,
    ``engine.predict`` -> ``ImageBatch`` — and asserts the batch carries the fields the
    export/crop instrument depends on. Raises LOUDLY (no silent fallback) on any drift.
    """
    from anomalib.engine import Engine

    from aiqs.data import build_datamodule

    dm = build_datamodule(cfg)
    model = build_model(cfg)
    eng = Engine(
        default_root_dir=str(Path(cfg.output.models_dir) / cfg.run_id),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_predict_batches=1,
    )
    eng.fit(model=model, datamodule=dm)
    preds = list(eng.predict(model=model, datamodule=dm, return_predictions=True) or [])
    if not preds:
        raise RuntimeError("smoke: engine.predict returned no batches.")
    batch = preds[0]
    missing = [f for f in ("pred_score", "image_path", "anomaly_map") if not hasattr(batch, f)]
    if missing:
        raise RuntimeError(
            f"smoke: ImageBatch is missing {missing} — anomalib 2.x API drift. Fix the field "
            f"names in _detector_v2.run_eval_export before the real run."
        )
    amap = np.asarray(_to_np(getattr(batch, "anomaly_map")[0])).squeeze()
    return ("smoke OK — build_datamodule + build_model + fit + predict ran; ImageBatch carries "
            f"pred_score / image_path / anomaly_map (map shape {amap.shape}).")


def _image_metrics(scores, labels) -> dict:
    out = {"pixel_auroc": None, "pixel_aupro": None, "pixel_aupimo": None,
           "image_auroc": None, "image_f1score": None}
    y = [v for v in labels if v is not None]
    if len(set(y)) < 2:                      # AUROC undefined without both classes
        return out
    from sklearn.metrics import f1_score, roc_auc_score

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    out["image_auroc"] = float(roc_auc_score(y, s))
    thr = float(np.median(s))                # report-only F1 at a neutral threshold
    out["image_f1score"] = float(f1_score(y, (s >= thr).astype(int), zero_division=0))
    return out


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        return [x]
    return list(x)
