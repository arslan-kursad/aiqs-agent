"""Detector (off-the-shelf): EfficientAD via anomalib, plus train/eval Engines.

Per the hard constraints we do NOT define any model architecture here — we only
configure anomalib's EfficientAd and its Engine.
"""

from __future__ import annotations

from pathlib import Path

from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from aiqs.config import Config


def build_model(cfg: Config) -> EfficientAd:
    if cfg.model.name != "efficient_ad":
        raise ValueError(
            f"Phase 0 supports model 'efficient_ad' only, got '{cfg.model.name}'."
        )
    return EfficientAd(
        imagenet_dir=cfg.model.imagenet_dir,
        model_size=cfg.model.size,   # 'small' | 'medium'
    )


def checkpoint_path(cfg: Config) -> Path:
    """Deterministic checkpoint location (so eval finds what train wrote)."""
    return Path(cfg.output.models_dir) / cfg.run_id / "weights" / "model.ckpt"


def _root_dir(cfg: Config) -> str:
    return str(Path(cfg.output.models_dir) / cfg.run_id)


def build_train_engine(cfg: Config) -> Engine:
    """Engine for training. max_steps drives the schedule (max_epochs=-1)."""
    return Engine(
        task=TaskType.SEGMENTATION,
        default_root_dir=_root_dir(cfg),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_steps=cfg.training.max_steps,
        max_epochs=-1,               # let max_steps govern
        logger=False,
    )


def _silence_visualization() -> None:
    """Neutralise anomalib's per-image visualization callback.

    The Engine unconditionally appends a `_VisualizationCallback` that renders and
    saves a prediction overlay per image via matplotlib. Phase 0 needs metrics +
    scores, not overlays, and on a CPU host this rendering dominates eval wall-time
    (~13 min -> ~2 min for 160 images). We no-op its hooks rather than fight the
    Engine's fixed callback list. Idempotent.
    """
    from anomalib.callbacks.visualizer import _VisualizationCallback

    def _noop(self, *args, **kwargs):
        return None

    for hook in ("on_test_batch_end", "on_test_end",
                 "on_predict_batch_end", "on_predict_end"):
        setattr(_VisualizationCallback, hook, _noop)


def build_eval_engine(cfg: Config, image_metrics, pixel_metrics,
                      callbacks=None) -> Engine:
    """Engine for evaluation with the requested metric collections."""
    _silence_visualization()
    return Engine(
        task=TaskType.SEGMENTATION,
        default_root_dir=_root_dir(cfg),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        callbacks=callbacks,
        logger=False,
    )
