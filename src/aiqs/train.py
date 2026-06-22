"""Train anomaly detector on one MVTec AD category.

    uv run aiqs-train --config configs/patchcore_cpu.yaml --category screw
    uv run aiqs-train --config configs/default.yaml --category screw --max-steps 10
"""

from __future__ import annotations

import argparse

from lightning.pytorch import seed_everything

from aiqs.config import add_common_args, config_from_args
from aiqs.data import build_datamodule
from aiqs.detector import build_model, build_train_engine, checkpoint_path, is_step_driven


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detector on one category.")
    add_common_args(parser)
    cfg = config_from_args(parser.parse_args())

    seed_everything(cfg.seed, workers=True)

    budget = (f"max_steps={cfg.training.max_steps}" if is_step_driven(cfg)
              else "1 epoch (coreset memory-bank)")
    print(f"[train] {cfg.run_id} | category={cfg.category} "
          f"| {budget} | accelerator={cfg.training.accelerator}")

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)
    engine = build_train_engine(cfg)

    engine.fit(model=model, datamodule=datamodule)

    ckpt = checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(ckpt))
    print(f"[train] done. checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
