#!/usr/bin/env python3
"""Phase-2B GPU-host runner — MVTec AD 2 detector round, paste-and-run.

WHERE THIS RUNS: a CUDA host (Colab/Kaggle) with the anomalib-2.x env from
``requirements-ad2.txt`` + ``pip install -e . --no-deps``. It REFUSES to run on the pinned
1.2 stack (anomalib major < 2) — that would silently exercise the wrong backend.

WHAT IT DOES (two modes):
  --smoke  : a cheap 2.x API SHAKE-OUT (1 train batch + 1 predict batch) — run this FIRST.
             Catches MVTecAD2/PatchCore/Engine/ImageBatch API drift in seconds, not 40 min
             into a real train. Expect to fix 1-2 things here, that's the point.
  (full)   : train -> eval (image_scores.csv + anomaly maps) -> Phase-1 decide -> substrate
             report (aiqs-vlm --mock). Surfaces THE THREE NUMBERS that answer "did the 2.x
             path work AND does this AD2 category give image-level substrate":
               image_auroc, ESCALATE∩good, n_dw.

    # on the GPU host:
    python scripts/run_ad2_gpu.py --category sheet_metal --smoke
    python scripts/run_ad2_gpu.py --category sheet_metal

Read image_auroc FIRST: ~0.97 => detector saturated, no substrate (standard-MVTec repeat);
lower => substrate candidate -> then ESCALATE∩good (>=15 to proceed, >=30 powered) AND n_dw.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/patchcore_ad2.yaml"


def _require_anomalib_2() -> None:
    from aiqs._anomalib_compat import anomalib_major

    major = anomalib_major()
    if major < 2:
        sys.exit(
            f"STOP: anomalib major = {major}, need >= 2. This runner is GPU-host-only — "
            "install requirements-ad2.txt in a fresh env (see that file's header). Do NOT "
            "run the AD2 path against the pinned 1.2 stack."
        )


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=capture)


def _image_auroc(results_dir: str) -> str:
    import pandas as pd

    path = Path(results_dir) / "metrics.csv"
    if not path.exists():
        return "n/a (metrics.csv not found)"
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.endswith("image_auroc")), None)
    return f"{df[col].iloc[-1]:.4f}" if col is not None and len(df) else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2B AD2 GPU-host runner.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--category", default="sheet_metal", help="an AD2 category (not screw/capsule)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--smoke", action="store_true", help="cheap 2.x API shake-out, then exit")
    args = ap.parse_args()

    _require_anomalib_2()

    if args.smoke:
        from aiqs import _detector_v2
        from aiqs.config import load_config

        cfg = load_config(args.config)
        cfg.category = args.category
        print(f"[smoke] 2.x API shake-out on category={args.category} ...", flush=True)
        print("[smoke] " + _detector_v2.smoke(cfg))
        print("[smoke] PASS — the 2.x path is wired. Re-run without --smoke for the real round.")
        return 0

    base = ["--config", args.config, "--category", args.category]
    # 1-3 stream to console (the user watches train/eval/decide live).
    for cmd in (["aiqs-train", *base], ["aiqs-eval", *base], ["aiqs-decide"]):
        r = _run(cmd)
        if r.returncode != 0:
            sys.exit(f"STOP: `{cmd[0]}` failed (rc={r.returncode}). Fix it before continuing.")

    # 4) substrate report — capture so we can extract the numbers; rc==2 == SubstrateError
    #    (ESCALATE∩good < 15) is a VALID "no substrate" outcome, not a crash.
    vlm = _run(["aiqs-vlm", "--mock"], capture=True)
    out = (vlm.stdout or "") + (vlm.stderr or "")
    print(out)
    if vlm.returncode not in (0, 2):
        sys.exit(f"STOP: `aiqs-vlm --mock` failed (rc={vlm.returncode}).")

    good = re.search(r"good=(\d+)", out) or re.search(r"ESCALATE∩good = (\d+)", out)
    n_dw = re.search(r"n_dw=(\d+)", out)
    print("\n" + "=" * 74)
    print("  THE THREE NUMBERS  (send these to Claude)")
    print("=" * 74)
    print(f"  image_auroc    = {_image_auroc(args.results_dir)}   "
          "(>~0.97 => detector saturated, NO substrate)")
    print(f"  ESCALATE∩good  = {good.group(1) if good else '(see VLM block / SUBSTRATE GUARD above)'}"
          "   (>=15 proceed, >=30 powered)")
    print(f"  n_dw           = {n_dw.group(1) if n_dw else '(see run0 line above)'}")
    if vlm.returncode == 2:
        print("  NOTE: substrate guard fired (ESCALATE∩good < 15) => this category has NO "
              "image-level substrate; try another AD2 category.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
