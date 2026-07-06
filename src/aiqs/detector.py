"""Detector (off-the-shelf) via anomalib — version-dispatched train/eval Engines.

Per the hard constraints we do NOT define any model architecture; we only configure
anomalib's off-the-shelf models (EfficientAd, PatchCore) and the Engine. The decision
layer is detector-agnostic (it only consumes the persisted scores).

Two anomalib stacks, dispatched on ``anomalib.__version__`` (they never coexist):
  * **1.2** (pinned local Intel-mac): the ``_v1`` bodies below. anomalib imports are
    LAZY (inside the functions) so this module imports cleanly under EITHER version —
    a top-level `from anomalib import TaskType` would crash on import under 2.x.
  * **2.x** (GPU-host ``ad2`` extra): ``_detector_v2.py`` (metrics via ``Evaluator``,
    ``ImageBatch`` outputs, anomaly-map export). GPU-host-verified; never run locally.

Version-agnostic helpers (paths / schedule) stay here directly.
"""

from __future__ import annotations

from pathlib import Path

from aiqs._anomalib_compat import anomalib_major
from aiqs.config import Config

# --------------------------------------------------------------------------- #
# Version-agnostic helpers (no anomalib import).
# --------------------------------------------------------------------------- #


def is_step_driven(cfg: Config) -> bool:
    """EfficientAD trains on a step schedule; PatchCore builds a memory bank in a
    single epoch (its own trainer_arguments set max_epochs=1)."""
    return cfg.model.name == "efficient_ad"


def checkpoint_path(cfg: Config) -> Path:
    """Deterministic checkpoint location (so eval finds what train wrote)."""
    return Path(cfg.output.models_dir) / cfg.run_id / "weights" / "model.ckpt"


def _root_dir(cfg: Config) -> str:
    return str(Path(cfg.output.models_dir) / cfg.run_id)


# --------------------------------------------------------------------------- #
# Dispatching builders.
# --------------------------------------------------------------------------- #


def build_model(cfg: Config):
    """Construct the configured anomalib model (no architecture defined here)."""
    if anomalib_major() >= 2:
        from aiqs import _detector_v2

        return _detector_v2.build_model(cfg)
    return _build_model_v1(cfg)


def build_train_engine(cfg: Config):
    """Engine for training (step-driven EfficientAD vs single-epoch PatchCore)."""
    if anomalib_major() >= 2:
        from aiqs import _detector_v2

        return _detector_v2.build_train_engine(cfg, _root_dir(cfg), is_step_driven(cfg))
    return _build_train_engine_v1(cfg)


def build_eval_engine(cfg: Config, image_metrics, pixel_metrics, callbacks=None):
    """Engine for evaluation with the requested metric collections (1.2 path).

    The 2.x path attaches metrics via an ``Evaluator`` on the model, not via Engine args,
    so under 2.x evaluation is driven by ``_detector_v2`` (see ``evaluate.main``); this
    builder is the 1.2 contract only.
    """
    if anomalib_major() >= 2:  # pragma: no cover - GPU host; evaluate.main dispatches earlier
        raise RuntimeError(
            "build_eval_engine is the anomalib-1.2 contract; on 2.x use "
            "_detector_v2.run_eval_export (evaluate.main dispatches there)."
        )
    return _build_eval_engine_v1(cfg, image_metrics, pixel_metrics, callbacks)


# --------------------------------------------------------------------------- #
# anomalib 1.2 backend (in-line; lazy imports).
# --------------------------------------------------------------------------- #


def _build_model_v1(cfg: Config):
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
    raise ValueError(
        f"Unsupported model '{name}'. Supported: 'efficient_ad', 'patchcore'."
    )


def _build_train_engine_v1(cfg: Config):
    from anomalib import TaskType
    from anomalib.engine import Engine

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
    """Neutralise anomalib 1.2's per-image visualization callback.

    The Engine unconditionally appends a ``_VisualizationCallback`` that renders a
    prediction overlay per image via matplotlib; on a CPU host this dominates eval
    wall-time (~13 min -> ~2 min for 160 images). We no-op its hooks. Idempotent.
    (anomalib 2.x reworked visualization — ``_detector_v2`` handles it on its own.)
    """
    from anomalib.callbacks.visualizer import _VisualizationCallback

    def _noop(self, *args, **kwargs):
        return None

    for hook in ("on_test_batch_end", "on_test_end",
                 "on_predict_batch_end", "on_predict_end"):
        setattr(_VisualizationCallback, hook, _noop)


def _build_eval_engine_v1(cfg: Config, image_metrics, pixel_metrics, callbacks=None):
    from anomalib import TaskType
    from anomalib.engine import Engine

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
