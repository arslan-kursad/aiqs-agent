#!/usr/bin/env python3
"""Generate the BLIND human-read instrument for the UNCLASSIFIED escapes (Phase-2B §9).

The pre-registered rule declared the escape labeling INADEQUATE (unclassified 45% > the
frozen 0.30 ceiling). The pre-registered supplementary adjudication (protocol in CLAUDE.md,
2026-07-06) is a BLIND human read: the reader sees only the second-look REASONING text — no
ground-truth label, no arm, no formal verdict, rows shuffled — and labels each PERCEPTION /
SEMANTIC / UNCLEAR. This script produces:

  * ``blind_read.csv``     — blind_id, reasoning, label(empty)  → the reader fills `label`.
  * ``blind_read_key.csv`` — blind_id -> (image, gt_label, arm-B verdict, run/item)  → used
                             ONLY AFTER labeling, to score. DO NOT open it before the read.

Deterministic (fixed shuffle seed) so the instrument is reproducible. It does NOT change any
machine result — the rule's "labeling inadequate" verdict stands; this is a separate,
pre-registered read that REFINES the perception-vs-semantic question a human must settle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aiqs.decide import _find_run_dir
from aiqs.vlm import reasoning_rules as rr

SEED = 20260706


def unclassified_escapes(df: pd.DataFrame) -> pd.DataFrame:
    """The (run, item) ARM-A escapes whose ARM-B pair classified UNCLASSIFIED — the exact
    subset the pre-registered rule could not resolve. Deduplicated to unique images (temp=0
    makes per-run reasoning near-identical; a human reads each distinct part once)."""
    a = df[df.arm == "A"].set_index(["run", "image_path"])
    b = df[df.arm == "B"].set_index(["run", "image_path"])
    rows = []
    for (run, path), arow in a.iterrows():
        escaped = arow["label"] == 1 and arow["final_decision"] == "pass"
        if not escaped:
            continue
        brow = b.loc[(run, path)]
        # Diffuse items have no crop (ARM-B identical) — excluded from classification.
        if not bool(brow["had_crop"]):
            continue
        label = rr.classify_escape(brow["vlm_verdict"], str(brow["vlm_reasoning"]),
                                   diffuse=False)
        if label == rr.UNCLASSIFIED:
            rows.append({"image_path": path, "run": run,
                        "arm_b_verdict": brow["vlm_verdict"],
                        "arm_b_reasoning": str(brow["vlm_reasoning"]),
                        "gt_label": int(arow["label"])})
    out = pd.DataFrame(rows)
    return out.drop_duplicates("image_path").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the blind human-read instrument.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    run_dir = _find_run_dir(Path(args.results_dir), args.run)
    df = pd.read_csv(run_dir / "vlm_crop_results.csv")
    unc = unclassified_escapes(df)
    if unc.empty:
        print("No unclassified escapes — nothing to read.")
        return

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(unc))
    unc = unc.iloc[order].reset_index(drop=True)
    unc["blind_id"] = [f"R{i:03d}" for i in range(len(unc))]

    blind = unc[["blind_id", "arm_b_reasoning"]].rename(
        columns={"arm_b_reasoning": "reasoning"})
    blind["label"] = ""                                     # human fills: perception|semantic|unclear
    blind.to_csv(run_dir / "blind_read.csv", index=False)

    key = unc[["blind_id", "image_path", "gt_label", "arm_b_verdict", "run"]]
    key.to_csv(run_dir / "blind_read_key.csv", index=False)

    print(f"  wrote blind_read.csv ({len(blind)} rows) + blind_read_key.csv -> {run_dir}")
    print("  Read blind_read.csv, fill `label` per row (perception|semantic|unclear), THEN "
          "join on blind_id with blind_read_key.csv to score. Do NOT open the key first.")


if __name__ == "__main__":
    main()
