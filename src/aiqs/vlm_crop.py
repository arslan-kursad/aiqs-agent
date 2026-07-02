"""Phase-2B Stage-3 entry point: the two-arm full-vs-crop controlled experiment.

    uv run aiqs-vlm-crop --run <id> --mock     # wiring smoke (no API, walled off)
    uv run aiqs-vlm-crop --run <id>            # real (needs ANTHROPIC_API_KEY + anomaly maps)

Same ESCALATE bucket, two arms:
  ARM-A  full image only        (the 2A configuration — doubles as a 2A replication)
  ARM-B  full image + high-res crop on the anomaly-map peak (the hypothesis under test)

ARM INDEPENDENCE (design condition, enforced here): every VLM call is a FRESH single-turn
``messages.create`` — arms never share a conversation, so ARM-B cannot be anchored by
ARM-A's answer. Each arm gets its own state objects, the SAME item order, and the SAME
per-run seed schedule, so the only difference between arms is the crop itself.

GROUND-PARAMETRIC, SINGLE-GROUND RUNS: ``--run`` selects the ground (capsules first;
macaroni1 only AFTER the capsules result — see CLAUDE.md, the macaroni1 trigger).

DIFFUSE items (no crop) stay in the bucket — ARM-B is byte-identical to ARM-A for them —
but are EXCLUDED from the perception/semantic classification (pre-registered).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from aiqs.config import CropConfig
from aiqs.crop import compute_crop
from aiqs.decide import LOCKED_COST, _find_run_dir, _load_scores
from aiqs.eval import crop_eval as ce
from aiqs.eval import vlm_eval as ve
from aiqs.eval.decision import cross_venn_abers, decide
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import AnthropicVLMBackend, MockVLMBackend
from aiqs.vlm.crop_fn import make_crop_fn
from aiqs.vlm.state import VLMState
from aiqs.vlm.substrate import SubstrateError, bucket_composition, substrate_guard
from aiqs.vlm_decide import build_bucket_states

RESULTS_CSV = "vlm_crop_results.csv"
MOCK_PREFIX = "mock_"
MOCK_BANNER = ("# ⚠️ MOCK two-arm smoke — wiring only, NOT real-data evidence\n\n")


def load_map_index(run_dir: Path) -> dict:
    """(parent_dir_name, stem) -> absolute map path, from the run's anomaly-map manifest.
    Keyed structurally (not by absolute path) so GPU-host paths match on any host."""
    manifest = run_dir / "anomaly_maps" / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No anomaly-map manifest at {manifest}. ARM-B needs the maps exported by the "
            "detector round (run aiqs-eval on the GPU host; maps are gitignored, so run "
            "this experiment WHERE THE MAPS LIVE — the Kaggle session).")
    df = pd.read_csv(manifest)
    return {(Path(p).parent.name, Path(p).stem): str(run_dir / m)
            for p, m in zip(df["image_path"], df["map_path"])}


def attach_maps_and_diffuse(states: list[VLMState], map_index: dict,
                            crop_cfg: CropConfig) -> list[bool]:
    """Set each state's ``anomaly_map_path`` + precompute the per-item DIFFUSE flag
    (deterministic, no API): diffuse items get no crop and are classification-excluded."""
    from PIL import Image

    diffuse_by_item: list[bool] = []
    for s in states:
        key = (Path(s.image_path).parent.name, Path(s.image_path).stem)
        map_path = map_index.get(key)
        if map_path is None:
            raise FileNotFoundError(
                f"No anomaly map for bucket item {s.image_path} (key={key}) — the manifest "
                "does not cover the bucket. STOP (a silent full-image fallback here would "
                "contaminate ARM-B).")
        s.anomaly_map_path = map_path
        result = compute_crop(np.load(map_path), Image.open(s.image_path).convert("RGB"),
                              crop_cfg)
        diffuse_by_item.append(result.diffuse)
    return diffuse_by_item


def _fresh(template: VLMState, *, with_map: bool) -> VLMState:
    """A NEW state per (arm, run, item) — arms never share objects; ARM-A carries no map."""
    return VLMState(
        image_path=template.image_path, detector_score=template.detector_score,
        detector_p=template.detector_p, label=template.label,
        anomaly_map_path=template.anomaly_map_path if with_map else None)


def run_two_arm(run_dir: Path, *, k: int, mock: bool, seed: int, folds: int,
                token_cost: float, lambda_grid, repo_root: Path,
                max_items: int | None, crop_cfg: CropConfig):
    scores, labels, paths = _load_scores(run_dir)
    p_cross, _, _ = cross_venn_abers(scores, labels, k=folds, seed=seed)
    decisions = decide(p_cross, LOCKED_COST)
    comp = bucket_composition(labels, decisions)
    warnings = substrate_guard(comp["escalate_good"])       # raises below HARD_MIN

    esc_mask = comp["escalate_mask"]
    template = build_bucket_states(scores, labels, paths, p_cross, esc_mask, repo_root)
    if max_items is not None:
        template = template[:max_items]

    diffuse_by_item = attach_maps_and_diffuse(template, load_map_index(run_dir), crop_cfg)

    bucket_scores = np.array([s.detector_score for s in template])
    bucket_labels = np.array([s.label for s in template], dtype=int)
    det_full = ve.detector_hard_decision(scores, labels, LOCKED_COST)
    det_call_bucket = det_full[np.where(esc_mask)[0][:len(template)]]

    def backend_for(arm: str, run_idx: int):
        if mock:
            # SAME seed schedule for both arms (symmetric); arms differ only by the crop.
            return MockVLMBackend(seed=seed + run_idx)
        return (AnthropicVLMBackend(crop_fn=make_crop_fn(crop_cfg)) if arm == "B"
                else AnthropicVLMBackend())

    def run_arm(arm: str) -> list[list]:
        with_map = arm == "B"
        per_run = []
        for r in range(k):
            backend = backend_for(arm, r)
            per_run.append([adjudicate(_fresh(s, with_map=with_map), backend,
                                       LOCKED_COST, lam=0.0) for s in template])
        return per_run

    states_a = run_arm("A")
    states_b = run_arm("B")

    result = ce.evaluate_two_arm(
        states_a, states_b, bucket_scores, bucket_labels, det_call_bucket, LOCKED_COST,
        token_cost=token_cost, lambda_grid=lambda_grid, diffuse_by_item=diffuse_by_item,
        warnings=warnings, seed=seed)
    return result, comp, states_a, states_b


# --------------------------------------------------------------------------- #
# Persistence + console
# --------------------------------------------------------------------------- #

def write_results(run_dir: Path, states_a, states_b, mock: bool) -> Path:
    rows = []
    for arm, per_run in (("A", states_a), ("B", states_b)):
        for r, states in enumerate(per_run):
            for s in states:
                rows.append({
                    "arm": arm, "run": r, "image_path": s.image_path, "label": s.label,
                    "detector_score": s.detector_score, "detector_p": s.detector_p,
                    "vlm_verdict": s.vlm_verdict, "vlm_conf": s.vlm_conf,
                    "vlm_reasoning": s.vlm_reasoning,          # audit trail for the rules
                    "final_decision": s.final_decision.value, "abstained": s.abstained,
                    "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
                    "had_crop": s.anomaly_map_path is not None})
    path = run_dir / ((MOCK_PREFIX if mock else "") + RESULTS_CSV)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def summary_lines(res: ce.CropExperiment, comp: dict) -> list[str]:
    p, c, st, rep = res.paired, res.classification, res.stability, res.replication
    L = ["## Two-arm full-vs-crop experiment (Phase 2B Stage 3)", "",
         f"- Bucket: {comp['escalate_total']} ({comp['escalate_good']} good, "
         f"{comp['escalate_defective']} defective); K={res.arm_a.k_runs}; "
         f"diffuse (no crop, excluded from classification): {res.n_diffuse}.",
         f"- **Escape rate** A(full) {p['escape_rate_a_mean']:.3f} -> B(+crop) "
         f"{p['escape_rate_b_mean']:.3f} (Δ={p['delta_mean']:+.3f}; fixed-by-crop "
         f"{p['fixed_by_crop']}, broken-by-crop {p['broken_by_crop']} run-item pairs).",
         f"- **Escape classification** (PRE-REGISTERED rules, frozen pre-run): "
         f"perception={c['perception']}, semantic={c['semantic']}, "
         f"unclassified={c['unclassified']} (rate {c['unclassified_rate']:.2f}; "
         f"adequate={c['labeling_adequate']}), diffuse-excluded={c['diffuse_excluded']}.",
         f"- **Stable-vs-flip** (A escapes): {st['stable_wrong']}/{st['escaped_ever']} "
         f"stable-wrong (fraction {st['stable_fraction']:.2f}) — stable-wrong escapes are "
         f"invisible to K-run agreement.",
         f"- **Error-independence** (pre-registered, powered): A: {res.arm_a.rule_stability}"
         f" | B: {res.arm_b.rule_stability}.",
         f"- **2A replication (ARM-A, descriptive):** good-rescue "
         f"{rep['good_rescue_rate_mean']:.2f}, defect-escape "
         f"{rep['defect_escape_rate_mean']:.2f}, confidence-separation AUC "
         f"{rep['confidence_separation_auc']:.2f} (~0.5 = does not separate).",
         f"- **Tokens/call** A in/out {res.tokens_a['tokens_in_mean']:.0f}/"
         f"{res.tokens_a['tokens_out_mean']:.0f} | B {res.tokens_b['tokens_in_mean']:.0f}/"
         f"{res.tokens_b['tokens_out_mean']:.0f}.", ""]
    if res.warnings:
        L += ["**Guard warnings:**"] + [f"- {w}" for w in res.warnings] + [""]
    return L


def print_console(res: ce.CropExperiment, comp: dict) -> None:
    print("=" * 70)
    print("  AIQS-Agent — Phase 2B Stage 3: full-vs-crop two-arm experiment")
    print("=" * 70)
    for line in summary_lines(res, comp):
        if line and not line.startswith("#"):
            print("  " + line.replace("**", ""))
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-3 two-arm full-vs-crop experiment.")
    ap.add_argument("--run", help="results/runs/<run_id> (default: latest)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--mock", action="store_true", help="scripted VLM, walled off")
    ap.add_argument("--k", type=int, default=ve.RUN_K)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token-cost", type=float, default=0.0)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    repo_root = results_dir.resolve().parent if results_dir.name == "results" else Path.cwd()
    run_dir = _find_run_dir(results_dir, args.run)

    if not args.mock and not os.getenv("ANTHROPIC_API_KEY"):
        ap.error("ANTHROPIC_API_KEY not set (use --mock for the wiring smoke).")

    try:
        res, comp, states_a, states_b = run_two_arm(
            run_dir, k=args.k, mock=args.mock, seed=args.seed, folds=args.folds,
            token_cost=args.token_cost, lambda_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            repo_root=repo_root, max_items=args.max_items, crop_cfg=CropConfig())
    except SubstrateError as e:
        print(f"\n[SUBSTRATE GUARD] {e}\n")
        raise SystemExit(2)

    csv_path = write_results(run_dir, states_a, states_b, args.mock)
    body = "\n".join(summary_lines(res, comp)) + "\n"
    if args.mock:
        (run_dir / (MOCK_PREFIX + "vlm_crop_summary.md")).write_text(MOCK_BANNER + body)
        print("  [MOCK] two-arm wiring smoke — NOT evidence.")
    else:
        with open(run_dir / "summary.md", "a") as f:
            f.write("\n" + body)
    print_console(res, comp)
    print(f"  artifacts -> {csv_path.name} in {run_dir}\n")


if __name__ == "__main__":
    main()
