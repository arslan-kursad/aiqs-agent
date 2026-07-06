"""aiqs-model-tier-report — the cost-scaling comparison across VLM model tiers.

    uv run aiqs-model-tier-report --run <id>

Scans a run dir for every non-mock ``vlm_crop_results*.csv`` (a Haiku rehearsal, the
claude-sonnet-4-6 headline, an ARM-C free-tier run, ...) and reconstructs the FULL
two-arm evaluation (``eval.crop_eval.evaluate_two_arm``) from each persisted CSV — the
SAME aggregation code the live run used (including the degeneracy guard added in
``eval.vlm_eval``), so verdict distribution, escape delta, the pre-registered P/S/U
classification, error-independence, and token cost are directly comparable across tiers.

Walled off: writes ``model_tier_report.md`` in the run dir, NEVER appended to
``summary.md`` — the canonical headline document stays sonnet-only.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aiqs.decide import LOCKED_COST, _find_run_dir, _load_scores
from aiqs.eval import crop_eval as ce
from aiqs.eval.decision import Decision, cost_optimal_threshold
from aiqs.vlm_crop import _ckpt_path, summary_lines

# Leading "*" is load-bearing: a mock run's filename is PREFIXED (``mock_vlm_crop_results
# ...``), so a pattern anchored at "vlm_crop_results" would silently match zero mock files
# (glob has no "contains" mode without a leading wildcard) — caught by a test.
RESULTS_GLOB = "*vlm_crop_results*.csv"


@dataclass
class _Row:
    """Lightweight stand-in reconstructed from a persisted CSV row — carries exactly the
    fields ``eval.crop_eval``/``eval.vlm_eval`` read off a VLMState, nothing more."""

    image_path: str
    label: int
    vlm_verdict: str
    vlm_conf: float
    p_vlm: float
    vlm_reasoning: str
    final_decision: Decision
    abstained: bool
    tokens_in: float | None = None
    tokens_out: float | None = None
    anomaly_map_path: str | None = None   # unused by crop_eval; kept for interface parity


def _none_if_nan(value):
    """pandas reads a missing numeric cell as NaN (a float), not None — but
    ``token_cost_line`` checks ``is not None`` to decide whether usage was captured, so an
    un-normalized NaN would be silently counted as "present" and poison the mean."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def discover_variants(run_dir: Path, include_mock: bool = False) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(run_dir / RESULTS_GLOB)))
    if not include_mock:
        paths = [p for p in paths if not p.name.startswith("mock_")]
    return paths


def _rows_from_df(df: pd.DataFrame, arm: str, item_order: list[str]) -> list[list[_Row]]:
    sub = df[df.arm == arm]
    k = int(sub["run"].max()) + 1
    per_run = []
    for r in range(k):
        run_df = sub[sub.run == r].set_index("image_path")
        rows = [_Row(
            image_path=path, label=int(run_df.loc[path, "label"]),
            vlm_verdict=run_df.loc[path, "vlm_verdict"],
            vlm_conf=float(run_df.loc[path, "vlm_conf"]),
            p_vlm=float(run_df.loc[path, "p_vlm"]),
            vlm_reasoning=str(run_df.loc[path, "vlm_reasoning"]),
            final_decision=Decision(run_df.loc[path, "final_decision"]),
            abstained=bool(run_df.loc[path, "abstained"]),
            tokens_in=_none_if_nan(run_df.loc[path].get("tokens_in")),
            tokens_out=_none_if_nan(run_df.loc[path].get("tokens_out")),
        ) for path in item_order]
        per_run.append(rows)
    return per_run


def _det_threshold(run_dir: Path) -> float:
    scores, labels, _ = _load_scores(run_dir)
    return cost_optimal_threshold(scores, labels, LOCKED_COST)


def _wall_clock_minutes(run_dir: Path, model: str, provider: str) -> float | None:
    ckpt = _ckpt_path(run_dir, mock=False, model=model, provider=provider)
    if not ckpt.exists():
        return None
    timestamps = []
    with open(ckpt) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "timestamp" in rec:
                    timestamps.append(rec["timestamp"])
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)) / 60.0


def evaluate_variant(csv_path: Path, run_dir: Path, det_threshold: float) -> dict:
    df = pd.read_csv(csv_path)
    item_order = df[(df.arm == "A") & (df.run == 0)]["image_path"].tolist()
    states_a = _rows_from_df(df, "A", item_order)
    states_b = _rows_from_df(df, "B", item_order)

    item_meta = df.drop_duplicates("image_path").set_index("image_path")
    bucket_labels = np.array([int(item_meta.loc[p, "label"]) for p in item_order])
    bucket_scores = np.array([float(item_meta.loc[p, "detector_score"]) for p in item_order])
    det_call_bucket = (bucket_scores >= det_threshold).astype(int)

    b0 = df[(df.arm == "B") & (df.run == 0)].set_index("image_path")
    diffuse_by_item = [not bool(b0.loc[p, "had_crop"]) for p in item_order]

    result = ce.evaluate_two_arm(
        states_a, states_b, bucket_scores, bucket_labels, det_call_bucket, LOCKED_COST,
        token_cost=0.0, lambda_grid=[0.0], diffuse_by_item=diffuse_by_item, seed=0)

    model = str(df["model"].iloc[0]) if "model" in df.columns else "unknown"
    provider = str(df["provider"].iloc[0]) if "provider" in df.columns else "anthropic"
    comp = {"escalate_total": len(bucket_labels),
            "escalate_good": int((bucket_labels == 0).sum()),
            "escalate_defective": int((bucket_labels == 1).sum())}
    verdict_dist = {arm: df[df.arm == arm]["vlm_verdict"].value_counts(normalize=True)
                    .to_dict() for arm in ("A", "B")}
    return {"csv": csv_path.name, "model": model, "provider": provider, "result": result,
           "comp": comp, "verdict_dist": verdict_dist,
           "wall_clock_min": _wall_clock_minutes(run_dir, model, provider)}


def build_report(run_dir: Path, include_mock: bool = False) -> list[dict]:
    variants = discover_variants(run_dir, include_mock)
    if not variants:
        return []
    thr = _det_threshold(run_dir)
    return [evaluate_variant(p, run_dir, thr) for p in variants]


def _fmt_verdict_dist(dist: dict) -> str:
    return ", ".join(f"{k}={v:.0%}" for k, v in dist.items())


def format_table(rows: list[dict]) -> str:
    header = (f"| {'model':<30} | {'provider':<16} | {'escape A':>8} | {'escape B':>8} "
             f"| {'Δ':>7} | {'P/S/U':<12} | {'indep A':<8} | {'indep B':<8} "
             f"| {'tok/call A':>12} | {'tok/call B':>12} | {'wall(min)':>9} |")
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for r in rows:
        res = r["result"]
        c = res.classification
        psu = f"{c['perception']}/{c['semantic']}/{c['unclassified']}"
        wc = f"{r['wall_clock_min']:.1f}" if r["wall_clock_min"] is not None else "n/a"
        indep_a = "YES" if res.arm_a.rule_stability.startswith("YES") else "NO"
        indep_b = "YES" if res.arm_b.rule_stability.startswith("YES") else "NO"
        ta = f"{res.tokens_a['tokens_in_mean']:.0f}/{res.tokens_a['tokens_out_mean']:.0f}"
        tb = f"{res.tokens_b['tokens_in_mean']:.0f}/{res.tokens_b['tokens_out_mean']:.0f}"
        lines.append(
            f"| {r['model']:<30} | {r['provider']:<16} "
            f"| {res.paired['escape_rate_a_mean']:>8.3f} "
            f"| {res.paired['escape_rate_b_mean']:>8.3f} | {res.paired['delta_mean']:>+7.3f} "
            f"| {psu:<12} | {indep_a:<8} | {indep_b:<8} | {ta:>12} | {tb:>12} | {wc:>9} |")
    return "\n".join(lines)


def format_detail_sections(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        parts.append(f"### {r['model']} (provider: {r['provider']}, source: {r['csv']})\n")
        parts.append(f"- **Verdict distribution** (rubber-stamp check): A: "
                    f"{_fmt_verdict_dist(r['verdict_dist']['A'])} | B: "
                    f"{_fmt_verdict_dist(r['verdict_dist']['B'])}")
        parts.extend(summary_lines(r["result"], r["comp"]))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Model-tier comparison across all VLM two-arm runs in one run dir.")
    ap.add_argument("--run", help="results/runs/<run_id> (default: latest)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--include-mock", action="store_true",
                    help="include mock_* files too (debugging only — NOT evidence)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    run_dir = _find_run_dir(results_dir, args.run)
    rows = build_report(run_dir, include_mock=args.include_mock)
    if not rows:
        print(f"No {RESULTS_GLOB} found in {run_dir} — nothing to compare yet.")
        return

    table = format_table(rows)
    print("\n" + "=" * 70)
    print("  AIQS-Agent — Phase 2B model-tier comparison")
    print("=" * 70)
    print(table)
    print("=" * 70 + "\n")

    body = ("# Model-tier comparison (Phase 2B) — NOT the canonical headline document\n\n"
            "One row per VLM tier tested on this bucket (Haiku rehearsal, the "
            "claude-sonnet-4-6 headline, any ARM-C free-tier run, ...). See summary.md "
            "for the canonical (sonnet-4-6) result; this file is a cross-tier comparison "
            "only.\n\n" + table + "\n\n## Per-model detail\n\n"
            + format_detail_sections(rows))
    out_path = run_dir / "model_tier_report.md"
    out_path.write_text(body)
    print(f"  report -> {out_path}\n")


if __name__ == "__main__":
    main()
