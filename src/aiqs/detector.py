"""Detector (off-the-shelf) via anomalib, plus train/eval Engines.

Per the hard constraints we do NOT define any model architecture here — we only
configure anomalib's off-the-shelf models (EfficientAd, PatchCore) and the Engine.
The decision layer is detector-agnostic (it only consumes the persisted scores), so
swapping detectors is a config + small build branch, nothing downstream.
"""

from __future__ import annotations

from pathlib import Path

from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore

from aiqs.config import Config


def build_model(cfg: Config):
    """Construct the configured anomalib model (no architecture defined here)."""
    name = cfg.model.name
    if name == "efficient_ad":
        return EfficientAd(
            imagenet_dir=cfg.model.imagenet_dir,
            model_size=cfg.model.size,   # 'small' | 'medium'
        )
    if name == "patchcore":
        return Patchcore(
            backbone=cfg.model.backbone,
            layers=cfg.model.layers,
            coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
            num_neighbors=cfg.model.num_neighbors,
        )
    raise ValueError(
        f"Unsupported model '{name}'. Supported: 'efficient_ad', 'patchcore'."
    )


def is_step_driven(cfg: Config) -> bool:
    """EfficientAD trains on a step schedule; PatchCore builds a memory bank in a
    single epoch (its own trainer_arguments set max_epochs=1)."""
    return cfg.model.name == "efficient_ad"


def checkpoint_path(cfg: Config) -> Path:
    """Deterministic checkpoint location (so eval finds what train wrote)."""
    return Path(cfg.output.models_dir) / cfg.run_id / "weights" / "model.ckpt"


def _root_dir(cfg: Config) -> str:
    return str(Path(cfg.output.models_dir) / cfg.run_id)


def build_train_engine(cfg: Config) -> Engine:
    """Engine for training.

    EfficientAD trains on a step schedule (max_steps governs, max_epochs=-1).
    PatchCore builds its coreset memory bank in a single epoch; steps are
    irrelevant (max_epochs=1, max_steps=-1).
    """
    if is_step_driven(cfg):
        return Engine(
            task=TaskType.SEGMENTATION,
            default_root_dir=_root_dir(cfg),
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices,
            max_steps=cfg.training.max_steps,
            max_epochs=-1,               # let max_steps govern
            logger=False,
        )
    # Epoch-driven models (PatchCore): single-epoch memory-bank build.
    return Engine(
        task=TaskType.SEGMENTATION,
        default_root_dir=_root_dir(cfg),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_steps=-1,
        max_epochs=1,
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
