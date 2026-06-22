"""MVTec AD datamodule construction (anomalib auto-downloads on first use).

Original MVTec AD only. MVTec AD 2 is not supported on the anomalib 1.2 line —
logged as a Phase-1 follow-up in CLAUDE.md.
"""

from __future__ import annotations

from anomalib import TaskType
from anomalib.data import MVTec

from aiqs.config import Config


def build_datamodule(cfg: Config) -> MVTec:
    """Build the MVTec datamodule for the configured category.

    task=SEGMENTATION keeps ground-truth masks available, which the pixel-level
    metrics (AUPRO / AUPIMO) require.
    """
    if cfg.dataset.name != "mvtec":
        raise ValueError(
            f"Phase 0 supports the 'mvtec' dataset only, got '{cfg.dataset.name}'."
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
