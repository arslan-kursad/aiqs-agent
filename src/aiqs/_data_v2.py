"""anomalib 2.x datamodules — original MVTec AD (``MVTecAD``) and MVTec AD 2 (``MVTecAD2``).

GPU-HOST-VERIFIED, NOT RUN ON THE LOCAL 1.2 STACK
-------------------------------------------------
anomalib 2.x has no x86_64-macOS torch wheel, so this module is never imported on the
pinned Intel-mac stack (``data.build_datamodule`` dispatches here only when
``anomalib.__version__`` >= 2, i.e. on a CUDA host with the ``ad2`` extra). It encodes
the verified 2.x API (``MVTec`` -> ``MVTecAD`` rename; new ``MVTecAD2``; ``task=`` removed;
image sizing via augmentations) but is validated on the GPU host, not here — do not assume
it ran clean locally.

MVTec AD 2 ground truth (Stage-2 substrate measurement uses this): only the **public test**
split ships masks/labels offline; ``test_private`` / ``test_private_mixed`` GT lives behind
the MVTec eval server. So our labelled metrics (ESCALATE∩good, n_dw, VLM correctness) run on
the public-test split. ``test_type`` selects it.

TO VERIFY on the GPU host (do not guess silently — log the actual values):
  * the exact ``MVTecAD2(test_type=...)`` accepted value for the GT-bearing public split;
  * whether ``MVTecAD2`` auto-downloads, or needs a prepared local path (CC BY-NC-SA;
    likely a Kaggle/registered download) -> if manual, add an AD2 ``prepare_data`` parallel.
"""

from __future__ import annotations

from aiqs.config import Config

#: AD2 test split that carries offline ground truth (see module docstring).
_AD2_PUBLIC_TEST = "public"


def build_datamodule(cfg: Config):
    name = cfg.dataset.name
    if name == "mvtec":
        return _build_mvtec_ad(cfg)
    if name in ("mvtec_ad2", "mvtecad2"):
        return _build_mvtec_ad2(cfg)
    raise ValueError(
        f"anomalib-2.x backend supports 'mvtec' (MVTecAD) or 'mvtec_ad2' (MVTecAD2); "
        f"got '{name}'."
    )


def _augmentations(cfg: Config):
    """Resize transform — anomalib 2.x moved image sizing out of the datamodule ctor
    into a torchvision-v2 augmentation pipeline. Kept tiny: just the eval resize."""
    from torchvision.transforms.v2 import Resize

    return Resize(tuple(cfg.dataset.image_size), antialias=True)


def _build_mvtec_ad(cfg: Config):
    from anomalib.data import MVTecAD  # 2.x rename of the removed 1.2 `MVTec`

    return MVTecAD(
        root=cfg.dataset.root,
        category=cfg.category,
        train_batch_size=cfg.dataset.train_batch_size,
        eval_batch_size=cfg.dataset.eval_batch_size,
        num_workers=cfg.dataset.num_workers,
        augmentations=_augmentations(cfg),
        seed=cfg.seed,
    )


def _build_mvtec_ad2(cfg: Config):
    from anomalib.data import MVTecAD2

    return MVTecAD2(
        root=cfg.dataset.root,
        category=cfg.category,
        train_batch_size=cfg.dataset.train_batch_size,
        eval_batch_size=cfg.dataset.eval_batch_size,
        num_workers=cfg.dataset.num_workers,
        augmentations=_augmentations(cfg),
        test_type=_AD2_PUBLIC_TEST,   # GT-bearing split (TO VERIFY: exact value on GPU host)
        seed=cfg.seed,
    )
