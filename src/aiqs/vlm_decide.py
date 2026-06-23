"""Phase-2A entry point: VLM second-look on the ESCALATE bucket.

    uv run aiqs-vlm --mock                 # wiring smoke (no API, no cost), latest run
    uv run aiqs-vlm --run <id> --k 5        # real second-look (needs ANTHROPIC_API_KEY)
    make vlm RUN=<id>

WITHOUT re-running the detector, it: calibrates the persisted image scores (cross
Venn-Abers), decides the NATIVE locked-matrix policy, isolates the ESCALATE bucket,
enforces the substrate guard, then runs the VLM second-look K times and evaluates
mechanism + error-independence + bidirectional value + break-even shift.

SCOPE NOTE: the screw run is a WIRING SMOKE (full-image-only, native-74). The headline
mechanism comes only from a crop-equipped HARD-category run; never quote the smoke.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from aiqs.decide import LOCKED_COST, REALISTIC_COST, _find_run_dir, _load_scores
from aiqs.eval.decision import Decision, cross_venn_abers, decide
from aiqs.eval import vlm_eval as ve
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import AnthropicVLMBackend, MockVLMBackend
from aiqs.vlm.state import VLMState
from aiqs.vlm.substrate import SubstrateError, bucket_composition, substrate_guard

VLM_RESULTS_CSV = "vlm_results.csv"
MOCK_RESULTS_CSV = "mock_vlm_results.csv"
MOCK_SUMMARY = "mock_vlm_summary.md"
MOCK_BANNER = ("# ⚠️ MOCK VLM SMOKE — wiring only, NOT real-data evidence\n\n"
               "These numbers come from a SCRIPTED mock VLM (no real model, no images). "
               "Their only purpose is to prove the Phase-2A pipeline is wired correctly. "
               "They say NOTHING about real performance and must never be quoted.\n\n")


def remap_path(p: str, repo_root: Path) -> str:
    """Map a persisted (possibly Colab) image path to a local one by re-rooting at the
    'datasets/' segment: /content/aiqs/datasets/mvtec/... -> <repo>/datasets/mvtec/..."""
    parts = Path(p).parts
    if "datasets" in parts:
        tail = Path(*parts[parts.index("datasets"):])
        return str(repo_root / tail)
    return p


def build_bucket_states(scores, labels, paths, p_cross, esc_mask, repo_root):
    """One VLMState per ESCALATE item (carrying detector score + calibrated p + label)."""
    states = []
    for i in np.where(esc_mask)[0]:
        img = remap_path(paths[i], repo_root) if paths is not None else f"item_{i}"
        states.append(VLMState(image_path=img, detector_score=float(scores[i]),
                               detector_p=float(p_cross[i]), label=int(labels[i])))
    return states


def run_vlm(run_dir: Path, *, k: int, mock: bool, seed: int, folds: int,
            token_cost: float, lambda_grid, repo_root: Path, max_items: int | None):
    scores, labels, paths = _load_scores(run_dir)

    # PRIMARY calibration + NATIVE locked policy (the regime with usable substrate).
    p_cross, _, _ = cross_venn_abers(scores, labels, k=folds, seed=seed)
    decisions = decide(p_cross, LOCKED_COST)
    comp = bucket_composition(labels, decisions)
    warnings = substrate_guard(comp["escalate_good"])   # raises if < HARD_MIN

    esc_mask = comp["escalate_mask"]
    states_template = build_bucket_states(scores, labels, paths, p_cross, esc_mask, repo_root)
    if max_items is not None:
        states_template = states_template[:max_items]

    # Pinned detector hard-call on the SAME bucket items (basis of "detector wrong").
    bucket_scores = np.array([s.detector_score for s in states_template])
    bucket_labels = np.array([s.label for s in states_template], dtype=int)
    det_full = ve.detector_hard_decision(scores, labels, LOCKED_COST)
    det_call_bucket = det_full[np.where(esc_mask)[0][:len(states_template)]]

    backend = MockVLMBackend(seed=seed) if mock else AnthropicVLMBackend()

    # K runs (nondeterminism). Mock is deterministic per seed; vary seed per run so a
    # noisy mock / real model exposes verdict variance.
    states_per_run = []
    for r in range(k):
        if mock:
            backend = MockVLMBackend(seed=seed + r)
        run_states = [adjudicate(VLMState(**{f: getattr(s, f) for f in
                                            ("image_path", "detector_score", "detector_p",
                                             "label", "anomaly_map_path")}),
                                 backend, LOCKED_COST, lam=0.0)
                      for s in states_template]
        states_per_run.append(run_states)

    result = ve.evaluate(states_per_run, bucket_scores, bucket_labels, det_call_bucket,
                         LOCKED_COST, cost_label="10/3/1", token_cost=token_cost,
                         lambda_grid=lambda_grid, warnings=warnings, seed=seed)
    return result, comp, states_per_run


# --------------------------------------------------------------------------- #
# Persistence + console
# --------------------------------------------------------------------------- #

def write_vlm_results(run_dir: Path, states_per_run, result: ve.VLMEval,
                      mock: bool) -> Path:
    rows = []
    for r, states in enumerate(states_per_run):
        for s in states:
            rows.append({"run": r, "image_path": s.image_path, "label": s.label,
                         "detector_score": s.detector_score, "detector_p": s.detector_p,
                         "vlm_verdict": s.vlm_verdict, "vlm_conf": s.vlm_conf,
                         "p_vlm": s.p_vlm, "final_decision": s.final_decision.value,
                         "abstained": s.abstained})
    path = run_dir / (MOCK_RESULTS_CSV if mock else VLM_RESULTS_CSV)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def summary_section(result: ve.VLMEval, comp: dict) -> list[str]:
    r = result
    band_lo, band_hi = min(r.eff_review_cost_band), max(r.eff_review_cost_band)
    L = ["## VLM second-look (Phase 2A) — ESCALATE bucket", "",
         "_First LLM component (claude-sonnet-4-6 vision), ESCALATE-only. p_vlm is "
         "PROVISIONAL/UNCALIBRATED in 2A (full VLM calibration = 2B). Mechanism before "
         "economics; the real test is error-INDEPENDENCE from the detector._", "",
         f"- Bucket: {comp['escalate_total']} escalated "
         f"({comp['escalate_good']} good, {comp['escalate_defective']} defective); "
         f"K={r.k_runs} runs.",
         f"- (a) VLM raw accuracy vs GT: **{r.accuracy_mean:.3f}** (mean over K).",
         f"- (b) Error-independence (PRE-REGISTERED rule, P_IND_MIN={ve.P_IND_MIN}, "
         f"KAPPA_MAX={ve.KAPPA_MAX}): **{r.rule_stability}**; "
         f"distribution {r.rule_distribution}.",
         f"- (c) Bidirectional value (mean/run): rescued→PASS "
         f"{r.bidirectional['rescued_to_pass']:.1f} (overkill↓), correct FAIL "
         f"{r.bidirectional['correct_fail']:.1f} (escape↓); VLM errors — wrong PASS "
         f"{r.bidirectional['wrong_pass_escape']:.1f}, wrong FAIL "
         f"{r.bidirectional['wrong_fail_overkill']:.1f}.",
         f"- (d) Effective review cost band over shrinkage λ∈"
         f"[{min(r.lambda_grid):g},{max(r.lambda_grid):g}]: "
         f"[{band_lo:.3f}, {band_hi:.3f}] vs bare review cost 1.000; Phase-1 native "
         f"break-even {ve.PHASE1_BREAKEVEN_NATIVE:.3f}. Break-even shifts RIGHT: "
         f"**{r.breakeven_shifts_right}**.", ""]
    if r.warnings:
        L += ["**Guard warnings:**"] + [f"- {w}" for w in r.warnings] + [""]
    L += ["**Caveat (2A = mechanism + DIRECTION, not magnitude).** Full-image-only on a "
          "walled-off smoke category; small n → wide CIs; headline mechanism awaits a "
          "crop-equipped hard-category run.", ""]
    return L


def append_summary(run_dir: Path, lines: list[str], mock: bool) -> None:
    body = "\n".join(lines) + "\n"
    if mock:
        # Walled off: a mock smoke NEVER touches the real summary.md (it is not evidence).
        (run_dir / MOCK_SUMMARY).write_text(MOCK_BANNER + body)
        return
    path = run_dir / "summary.md"
    marker = "## VLM second-look (Phase 2A)"
    if path.exists():
        existing = path.read_text()
        if marker in existing:
            existing = existing.split(marker)[0].rstrip()
        path.write_text(existing + "\n\n" + body)
    else:
        path.write_text(body)


def print_console(result: ve.VLMEval, comp: dict) -> None:
    r = result
    print("=" * 66)
    print("  AIQS-Agent — Phase 2A: VLM second-look (ESCALATE bucket)")
    print("=" * 66)
    print(f"  bucket={comp['escalate_total']}  good={comp['escalate_good']}  "
          f"defective={comp['escalate_defective']}  K={r.k_runs}")
    for w in r.warnings:
        print(f"  [warn] {w}")
    print(f"  (a) VLM accuracy (mean/K)        = {r.accuracy_mean:.3f}")
    print(f"  (b) error-independence rule      = {r.rule_stability}")
    print(f"      distribution                 = {r.rule_distribution}")
    o0 = r.rule_outcomes[0]
    print(f"      run0: P(VLM ok|det wrong)={o0.p_ind_point:.3f}"
          f" [lo={o0.p_ind_lo:.3f}] n_dw={o0.n_detector_wrong}  κ={o0.kappa:.3f}"
          f" [{o0.kappa_lo:.3f},{o0.kappa_hi:.3f}]")
    print(f"  (c) rescued→PASS={r.bidirectional['rescued_to_pass']:.1f}  "
          f"correct FAIL={r.bidirectional['correct_fail']:.1f}  "
          f"(wrong PASS={r.bidirectional['wrong_pass_escape']:.1f})")
    band_lo, band_hi = min(r.eff_review_cost_band), max(r.eff_review_cost_band)
    print(f"  (d) eff. review cost band        = [{band_lo:.3f}, {band_hi:.3f}] "
          f"(bare review=1.000); break-even shifts right: {r.breakeven_shifts_right}")
    print("=" * 66 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2A VLM second-look (ESCALATE bucket).")
    parser.add_argument("--run", help="results/runs/<run_id> (default: latest).")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--mock", action="store_true",
                        help="use the mock VLM (no API, no cost) — wiring smoke.")
    parser.add_argument("--k", type=int, default=ve.RUN_K, help="repeat runs (default 5).")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-cost", type=float, default=0.0,
                        help="VLM token cost per item in matrix-relative units (default 0).")
    parser.add_argument("--max-items", type=int, default=None,
                        help="cap bucket size (debugging / cost control).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    repo_root = results_dir.resolve().parent if results_dir.name == "results" \
        else Path.cwd()
    run_dir = _find_run_dir(results_dir, args.run)

    if not args.mock and not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY not set. Use --mock for the wiring smoke, or set "
                     "the key in .env for a real run.")

    lambda_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    try:
        result, comp, states_per_run = run_vlm(
            run_dir, k=args.k, mock=args.mock, seed=args.seed, folds=args.folds,
            token_cost=args.token_cost, lambda_grid=lambda_grid, repo_root=repo_root,
            max_items=args.max_items)
    except SubstrateError as e:
        print(f"\n[SUBSTRATE GUARD] {e}\n")
        raise SystemExit(2)

    csv_path = write_vlm_results(run_dir, states_per_run, result, args.mock)
    append_summary(run_dir, summary_section(result, comp), args.mock)
    if args.mock:
        print("  [MOCK] wiring smoke — scripted VLM, NOT real-data evidence.")
    print_console(result, comp)
    summary_name = MOCK_SUMMARY if args.mock else "summary.md (VLM section)"
    print(f"  artifacts -> {csv_path.name}, {summary_name} in {run_dir}\n")


if __name__ == "__main__":
    main()
