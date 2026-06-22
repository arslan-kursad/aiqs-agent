"""Fetch one MVTec AD category and lay it out the way anomalib expects.

Why this exists: anomalib 1.2.0 hard-codes a MVTec AD download URL (an old
mydrive.ch share) that now returns HTTP 404. Rather than pull the full 4.9 GB
dataset from a flaky source, we fetch ONLY the requested category from a public
Hugging Face mirror and reorganise it into anomalib's native folder layout. Then
anomalib's own existence check (`<root>/<category>` is a dir) skips its dead
download path entirely.

Mirror layout (TheoM55/mvtec_anomaly_detection, split/category transposed):
    images/train/<cat>/good/*.png
    images/test/<cat>/<defect>/*.png
    masks/test/<cat>/<defect>/<id>_mask.png

anomalib layout we produce (under <root>/<cat>/):
    train/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/<id>_mask.png   (masks already named *_mask.png)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MIRROR_REPO = "TheoM55/mvtec_anomaly_detection"


def prepare_mvtec(category: str, root: str | Path = "datasets/mvtec",
                  repo_id: str = MIRROR_REPO) -> Path:
    """Ensure <root>/<category> exists in anomalib's MVTec layout. Idempotent."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    target = root / category
    if (target / "train" / "good").is_dir() and (target / "ground_truth").is_dir():
        print(f"[data] {category}: already present at {target} — skipping.")
        return target

    staging = root / f"_staging_{category}"
    print(f"[data] {category}: fetching from {repo_id} (category only) ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            f"images/train/{category}/**",
            f"images/test/{category}/**",
            f"masks/test/{category}/**",
        ],
        local_dir=str(staging),
    )

    src_train = staging / "images" / "train" / category
    src_test = staging / "images" / "test" / category
    src_masks = staging / "masks" / "test" / category
    if not src_train.is_dir() or not src_test.is_dir():
        raise FileNotFoundError(
            f"Mirror {repo_id} did not yield category '{category}' "
            f"(looked for {src_train} and {src_test})."
        )

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_train, target / "train", dirs_exist_ok=True)
    shutil.copytree(src_test, target / "test", dirs_exist_ok=True)
    if src_masks.is_dir():
        shutil.copytree(src_masks, target / "ground_truth", dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"No ground-truth masks for '{category}' in {repo_id}.")

    shutil.rmtree(staging, ignore_errors=True)
    _report(target)
    return target


def _report(target: Path) -> None:
    train_good = len(list((target / "train" / "good").glob("*.png")))
    test_defects = sorted(p.name for p in (target / "test").iterdir() if p.is_dir())
    gt_defects = sorted(p.name for p in (target / "ground_truth").iterdir() if p.is_dir())
    n_test = len(list((target / "test").rglob("*.png")))
    print(f"[data] ready: {target}")
    print(f"[data]   train/good: {train_good} imgs | test: {n_test} imgs "
          f"across {test_defects}")
    print(f"[data]   ground_truth defects: {gt_defects}")


def make_synthetic_imagenette(path: str | Path, n_per_class: int = 8,
                              classes: int = 2, size: int = 256) -> Path:
    """Create a tiny ImageFolder-shaped stand-in for ImageNette.

    Used ONLY by the smoke test so EfficientAD training runs without the ~1.5 GB
    real ImageNette download. Content is random noise — it validates wiring, not
    model quality. Real baselines must use the real ImageNette.
    """
    import numpy as np
    from PIL import Image

    path = Path(path)
    if path.is_dir() and any(path.iterdir()):
        print(f"[data] synthetic imagenette already at {path} — skipping.")
        return path

    rng = np.random.default_rng(0)
    for c in range(classes):
        cls_dir = path / f"n{c:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, size=(size, size, 3), dtype="uint8")
            Image.fromarray(arr).save(cls_dir / f"{i:03d}.png")
    print(f"[data] wrote synthetic imagenette ({classes}x{n_per_class} imgs) -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase-0 datasets.")
    parser.add_argument("--category", default="screw", help="MVTec AD category.")
    parser.add_argument("--root", default="datasets/mvtec", help="Dataset root.")
    parser.add_argument("--repo-id", default=MIRROR_REPO, help="HF mirror repo.")
    parser.add_argument(
        "--make-synthetic-imagenette", metavar="DIR",
        help="Instead of MVTec, write a tiny synthetic ImageNette here (smoke test).",
    )
    args = parser.parse_args()
    if args.make_synthetic_imagenette:
        make_synthetic_imagenette(args.make_synthetic_imagenette)
    else:
        prepare_mvtec(args.category, args.root, args.repo_id)


if __name__ == "__main__":
    main()
