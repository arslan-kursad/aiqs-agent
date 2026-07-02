"""Typed config: YAML file + CLI overrides.

Deliberately small (stdlib + PyYAML, nested dataclasses). No omegaconf/pydantic —
Phase 0 does not need them, and the north star says no premature abstraction.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str = "mvtec"
    root: str = "datasets/mvtec"
    image_size: tuple[int, int] = (256, 256)
    train_batch_size: int = 1
    eval_batch_size: int = 16
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "efficient_ad"          # 'efficient_ad' | 'patchcore'
    size: str = "small"                 # EfficientAD only: small | medium
    # EfficientAD's penalty/distillation dataset. Real training pulls the full
    # ImageNette (~1.5 GB). The smoke test points this at a tiny synthetic folder
    # to keep wiring validation fast (see prepare_data.make_synthetic_imagenette).
    imagenet_dir: str = "datasets/imagenette"
    # PatchCore-only knobs (ignored by EfficientAD). PatchCore builds a coreset
    # memory bank in a single pass (no step grind) and tends to separate better at
    # image level — see CLAUDE.md. A light backbone + small coreset keep it CPU-cheap.
    backbone: str = "wide_resnet50_2"
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9

    @property
    def variant(self) -> str:
        """Short label distinguishing artifacts (EfficientAD size vs PatchCore backbone)."""
        return self.size if self.name == "efficient_ad" else self.backbone


@dataclass
class TrainingConfig:
    max_steps: int = 70000
    accelerator: str = "auto"
    devices: int = 1


@dataclass
class OutputConfig:
    models_dir: str = "models"
    results_dir: str = "results"


@dataclass
class CropConfig:
    """Phase-2B anomaly-map crop instrument (see src/aiqs/crop.py).

    ``enabled`` is the full-image-only (False) vs full-image+crop (True) switch for the
    Stage-3 two-arm experiment. All thresholds are detector-free knobs tuned offline.
    """
    enabled: bool = False
    peak_top_frac: float = 0.01           # peak candidates = top this fraction of map pixels
    peak_fraction: float = 0.5            # floor on the normalized threshold (guards sparse maps)
    padding: float = 0.25                 # bbox padding as a fraction of the bbox size
    min_size: int = 64                    # minimum crop side, in ORIGINAL-image pixels
    diffuse_area_frac: float = 0.5        # peak-region BBOX over this frac of the frame -> diffuse


@dataclass
class Config:
    seed: int = 42
    category: str = "screw"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    crop: CropConfig = field(default_factory=CropConfig)

    # ---- run identity (derived) -------------------------------------------
    @property
    def run_id(self) -> str:
        """Stable, human-readable id used to name model/result artifacts."""
        return (f"{self.model.name}-{self.model.variant}"
                f"_{self.dataset.name}-{self.category}")

    def as_flat_dict(self) -> dict:
        """Flattened view for logging / CSV run-metadata columns."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "category": self.category,
            "dataset": self.dataset.name,
            "model": self.model.name,
            "model_size": self.model.variant,
            "image_size": f"{self.dataset.image_size[0]}x{self.dataset.image_size[1]}",
            "max_steps": self.training.max_steps,
            "accelerator": self.training.accelerator,
        }


def _build_nested(cls, data: dict):
    """Recursively coerce a plain dict into a (nested) dataclass instance.

    Nested sub-configs are detected via each field's ``default_factory`` (a
    dataclass type), which is robust to stringised annotations from
    ``from __future__ import annotations``.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in (data or {}).items():
        if key not in field_map:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        factory = field_map[key].default_factory
        is_subconfig = (
            factory is not dataclasses.MISSING
            and isinstance(factory, type)
            and dataclasses.is_dataclass(factory)
        )
        if is_subconfig and isinstance(value, dict):
            kwargs[key] = _build_nested(factory, value)
        elif key == "image_size" and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build_nested(Config, raw)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags for train/eval entry points."""
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config."
    )
    parser.add_argument("--category", help="Override MVTec AD category.")
    parser.add_argument(
        "--max-steps", type=int, help="Override training.max_steps (e.g. smoke test)."
    )
    parser.add_argument("--accelerator", help="Override training.accelerator.")
    parser.add_argument("--imagenet-dir", help="Override model.imagenet_dir.")


def config_from_args(args: argparse.Namespace) -> Config:
    """Load YAML then apply any CLI overrides that were provided."""
    cfg = load_config(args.config)
    if getattr(args, "category", None):
        cfg.category = args.category
    if getattr(args, "max_steps", None) is not None:
        cfg.training.max_steps = args.max_steps
    if getattr(args, "accelerator", None):
        cfg.training.accelerator = args.accelerator
    if getattr(args, "imagenet_dir", None):
        cfg.model.imagenet_dir = args.imagenet_dir
    return cfg
