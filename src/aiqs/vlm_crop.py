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
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aiqs.config import CropConfig
from aiqs.crop import compute_crop
from aiqs.decide import LOCKED_COST, _find_run_dir, _load_scores
from aiqs.eval import crop_eval as ce
from aiqs.eval import vlm_eval as ve
from aiqs.eval.decision import Decision, cross_venn_abers, decide
from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import MODEL, AnthropicVLMBackend, MockVLMBackend, VLMParseError
from aiqs.vlm.crop_fn import make_crop_fn
from aiqs.vlm.state import VLMState
from aiqs.vlm.substrate import SubstrateError, bucket_composition, substrate_guard
from aiqs.vlm_decide import build_bucket_states

RESULTS_CSV = "vlm_crop_results.csv"
CHECKPOINT = "vlm_crop_checkpoint.jsonl"
MOCK_PREFIX = "mock_"
MOCK_BANNER = ("# ⚠️ MOCK two-arm smoke — wiring only, NOT real-data evidence\n\n")
PROGRESS_EVERY = 25


# --------------------------------------------------------------------------- #
# Checkpoint / resume — the money-protection layer.
#
# Every completed (arm, run, item) call is appended to a JSONL checkpoint the moment it
# returns, so ANY mid-run failure (529 storm past the retry budget, a parse error, a
# served-model stop, a dead kernel) loses AT MOST ONE call. Re-running the same command
# RESUMES: completed calls are reconstructed from disk and never re-billed. The file is
# also the raw audit trail (verdict + reasoning + tokens per call).
# --------------------------------------------------------------------------- #

def _model_suffix(model: str, provider: str = "anthropic") -> str:
    """Non-canonical models (anything but the LOCKED claude-sonnet-4-6 on the anthropic
    provider) get their own artifact namespace, so a cheap-model rehearsal OR a free-tier
    ARM-C run can NEVER contaminate (or be resumed into) the canonical run — separate
    checkpoint, separate results, separate summary.

    Non-anthropic providers are namespaced by BOTH provider and model (``__provider__model``)
    since two different free-tier endpoints could plausibly reuse a similar model string —
    the canonical anthropic/claude-sonnet-4-6 suffix behaviour is UNCHANGED (empty string),
    so every existing checkpoint/results file keeps resolving exactly as before.
    """
    if model == MODEL and provider == "anthropic":
        return ""
    tag = model if provider == "anthropic" else f"{provider}__{model}"
    return f"__{tag}"


def _ckpt_path(run_dir: Path, mock: bool, model: str = MODEL,
              provider: str = "anthropic") -> Path:
    return run_dir / ((MOCK_PREFIX if mock else "")
                      + CHECKPOINT.replace(".jsonl", _model_suffix(model, provider) + ".jsonl"))


def load_checkpoint(path: Path) -> dict:
    done: dict = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[(rec["arm"], rec["run"], rec["idx"])] = rec
    return done


def _record(arm: str, r: int, i: int, s: VLMState, model: str, provider: str) -> dict:
    return {"arm": arm, "run": r, "idx": i, "image_path": s.image_path, "model": model,
            "provider": provider, "timestamp": time.time(),
            "vlm_verdict": s.vlm_verdict, "vlm_conf": s.vlm_conf,
            "vlm_reasoning": s.vlm_reasoning, "p_vlm": s.p_vlm,
            "final_decision": s.final_decision.value, "abstained": s.abstained,
            "tokens_in": s.tokens_in, "tokens_out": s.tokens_out}


def _restore(rec: dict, template: VLMState, with_map: bool, model: str,
            provider: str = "anthropic") -> VLMState:
    if rec.get("model", model) != model or rec.get("provider", "anthropic") != provider:
        raise RuntimeError(
            f"CHECKPOINT MISMATCH: checkpoint records are from "
            f"(provider={rec.get('provider', 'anthropic')!r}, model={rec.get('model')!r}) "
            f"but this run requests (provider={provider!r}, model={model!r}) — refusing to "
            "mix models/providers in one experiment.")
    if rec["image_path"] != template.image_path:
        raise RuntimeError(
            f"CHECKPOINT MISMATCH at (arm={rec['arm']}, run={rec['run']}, idx={rec['idx']}): "
            f"checkpoint has {rec['image_path']}, bucket has {template.image_path}. The "
            "checkpoint belongs to a DIFFERENT bucket — delete it only if you know why.")
    s = _fresh(template, with_map=with_map)
    s.vlm_verdict, s.vlm_conf = rec["vlm_verdict"], rec["vlm_conf"]
    s.vlm_reasoning, s.p_vlm = rec["vlm_reasoning"], rec["p_vlm"]
    s.final_decision = Decision(rec["final_decision"])
    s.abstained = rec["abstained"]
    s.tokens_in, s.tokens_out = rec.get("tokens_in"), rec.get("tokens_out")
    return s


def _adjudicate_loud_fallback(state: VLMState, backend, cost) -> tuple[VLMState, bool]:
    """One call with a bounded parse-failure fallback (LOUD, never silent, never blocking).

    temperature=0 means a malformed response can be DETERMINISTIC for an item — a hard
    raise would make the run un-completable (resume hits the same wall forever). So: retry
    once; if still malformed, mark the item ``unsure`` with a PARSE_FAILURE-prefixed
    reasoning (-> abstain/ESCALATE, the conservative outcome; classified UNCLASSIFIED),
    print a warning, and report the count. Returns (state, parse_failed)."""
    for attempt in (1, 2):
        try:
            return adjudicate(state, backend, cost, lam=0.0), False
        except VLMParseError as e:
            last = e
            print(f"  [warn] parse failure (attempt {attempt}/2) on {state.image_path}")
    state.vlm_verdict, state.vlm_conf = "unsure", 0.0
    state.vlm_reasoning = f"PARSE_FAILURE: {str(last)[:200]}"
    state.p_vlm = confidence_to_p("unsure", 0.0)
    state.final_decision = adjudicate_probability(state.p_vlm, cost)
    state.abstained = state.final_decision is Decision.ESCALATE
    return state, True


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


def crop_fn_for(provider: str, crop_cfg: CropConfig):
    """The SAME crop instrument (``aiqs.crop.compute_crop``) for every provider — only the
    content-block WRAPPING differs (Anthropic vs OpenAI-compatible image-block shape)."""
    if provider == "openai_compatible":
        from aiqs.vlm.backend_openai_compatible import OpenAICompatibleVLMBackend

        return make_crop_fn(crop_cfg, encode_fn=OpenAICompatibleVLMBackend._encode_image_block)
    return make_crop_fn(crop_cfg)


def build_backend(provider: str, model: str, *, crop_fn=None, base_url: str | None = None,
                  api_key_env: str | None = None, rpm_limit: float | None = None):
    """Construct the real (non-mock) backend for a provider. A single dispatch point so
    ``run_two_arm`` and ``run_smoke`` never diverge on how a backend is built."""
    if provider == "anthropic":
        return AnthropicVLMBackend(model=model, crop_fn=crop_fn)
    if provider == "openai_compatible":
        from aiqs.vlm.backend_openai_compatible import OpenAICompatibleVLMBackend

        if not base_url or not api_key_env:
            raise ValueError("provider='openai_compatible' requires --base-url and "
                             "--api-key-env.")
        return OpenAICompatibleVLMBackend(model=model, base_url=base_url,
                                          api_key_env=api_key_env, rpm_limit=rpm_limit,
                                          crop_fn=crop_fn)
    raise ValueError(f"Unsupported provider {provider!r}. Use 'anthropic' or "
                     "'openai_compatible'.")


def run_smoke(run_dir: Path, *, model: str, provider: str, base_url: str | None,
             api_key_env: str | None, rpm_limit: float | None, repo_root: Path,
             crop_cfg: CropConfig, seed: int = 42, folds: int = 10) -> None:
    """ONE real call per arm (2 total) to shake out a new provider/endpoint BEFORE a real
    paid run: confirms auth, vision-content acceptance, the served-model string, and that
    usage fields populate — never assumed. Writes to a dedicated *_smoke.jsonl file, kept
    OUT of the real checkpoint/resume path so it can never be mistaken for a paid call."""
    scores, labels, paths = _load_scores(run_dir)
    p_cross, _, _ = cross_venn_abers(scores, labels, k=folds, seed=seed)
    decisions = decide(p_cross, LOCKED_COST)
    comp = bucket_composition(labels, decisions)
    esc_mask = comp["escalate_mask"]
    template = build_bucket_states(scores, labels, paths, p_cross, esc_mask, repo_root)[:1]
    if not template:
        raise SubstrateError("Bucket is empty — cannot smoke-test.")
    attach_maps_and_diffuse(template, load_map_index(run_dir), crop_cfg)

    print(f"  [smoke] provider={provider} model={model} — 1 real call per arm ...")
    for arm, with_map in (("A", False), ("B", True)):
        cfn = crop_fn_for(provider, crop_cfg) if with_map else None
        backend = build_backend(provider, model, crop_fn=cfn, base_url=base_url,
                                api_key_env=api_key_env, rpm_limit=rpm_limit)
        state = _fresh(template[0], with_map=with_map)
        verdict = backend(state)
        print(f"  [smoke][ARM-{arm}] served ok | verdict={verdict.verdict} "
              f"conf={verdict.confidence:.2f} | tokens_in={state.tokens_in} "
              f"tokens_out={state.tokens_out} (None => usage field not populated — check "
              "the provider's response shape before the real run)")
    print("  [smoke] PASS — auth + served-model + content acceptance confirmed for both "
          "arms. Re-run without --smoke for the real round.")


def run_two_arm(run_dir: Path, *, k: int, mock: bool, seed: int, folds: int,
                token_cost: float, lambda_grid, repo_root: Path,
                max_items: int | None, crop_cfg: CropConfig, model: str = MODEL,
                provider: str = "anthropic", base_url: str | None = None,
                api_key_env: str | None = None, rpm_limit: float | None = None):
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
        cfn = crop_fn_for(provider, crop_cfg) if arm == "B" else None
        return build_backend(provider, model, crop_fn=cfn, base_url=base_url,
                            api_key_env=api_key_env, rpm_limit=rpm_limit)

    ckpt = _ckpt_path(run_dir, mock, model, provider)
    done = load_checkpoint(ckpt)
    total = 2 * k * len(template)
    print(f"  [plan] {len(template)} items x 2 arms x K={k} = {total} calls; "
          f"resumed from checkpoint: {len(done)}; new calls this session: "
          f"{total - len(done)}  (checkpoint: {ckpt.name})")

    t0 = time.time()
    parse_failures = 0
    calls_new = 0

    def run_arm(arm: str) -> list[list]:
        nonlocal parse_failures, calls_new
        with_map = arm == "B"
        per_run = []
        with open(ckpt, "a") as f:
            for r in range(k):
                backend = backend_for(arm, r)
                states = []
                for i, s in enumerate(template):
                    key = (arm, r, i)
                    if key in done:                     # already paid for — restore, skip
                        states.append(_restore(done[key], s, with_map, model, provider))
                        continue
                    state, failed = _adjudicate_loud_fallback(
                        _fresh(s, with_map=with_map), backend, LOCKED_COST)
                    parse_failures += failed
                    f.write(json.dumps(_record(arm, r, i, state, model, provider)) + "\n")
                    f.flush()                            # a crash now loses ZERO calls
                    states.append(state)
                    calls_new += 1
                    if calls_new % PROGRESS_EVERY == 0:
                        el = (time.time() - t0) / 60
                        print(f"  [{arm} run {r}] {calls_new}/{total - len(done)} new calls "
                              f"| {el:.1f} min elapsed", flush=True)
                per_run.append(states)
        return per_run

    states_a = run_arm("A")
    states_b = run_arm("B")
    if parse_failures:
        print(f"  [warn] {parse_failures} call(s) fell back to unsure after repeated parse "
              "failures (marked PARSE_FAILURE in reasoning/results — inspect them).")

    result = ce.evaluate_two_arm(
        states_a, states_b, bucket_scores, bucket_labels, det_call_bucket, LOCKED_COST,
        token_cost=token_cost, lambda_grid=lambda_grid, diffuse_by_item=diffuse_by_item,
        warnings=warnings, seed=seed)
    return result, comp, states_a, states_b


# --------------------------------------------------------------------------- #
# Persistence + console
# --------------------------------------------------------------------------- #

def write_results(run_dir: Path, states_a, states_b, mock: bool,
                  model: str = MODEL, provider: str = "anthropic") -> Path:
    rows = []
    for arm, per_run in (("A", states_a), ("B", states_b)):
        for r, states in enumerate(per_run):
            for s in states:
                rows.append({
                    "arm": arm, "run": r, "model": model, "provider": provider,
                    "image_path": s.image_path, "label": s.label,
                    "detector_score": s.detector_score, "detector_p": s.detector_p,
                    "vlm_verdict": s.vlm_verdict, "vlm_conf": s.vlm_conf, "p_vlm": s.p_vlm,
                    "vlm_reasoning": s.vlm_reasoning,          # audit trail for the rules
                    "final_decision": s.final_decision.value, "abstained": s.abstained,
                    "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
                    "had_crop": s.anomaly_map_path is not None})
    path = run_dir / ((MOCK_PREFIX if mock else "")
                      + RESULTS_CSV.replace(".csv", _model_suffix(model, provider) + ".csv"))
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
    ap.add_argument("--model", default=MODEL,
                    help=f"VLM model (default: the LOCKED {MODEL}). Any other value/provider "
                         "is a REHEARSAL: separate checkpoint/results namespace, never "
                         "appended to summary.md, never headline evidence.")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai_compatible"],
                    help="'anthropic' (default, the locked headline path) or "
                         "'openai_compatible' — ARM-C / the model-tier lever. Swapping a "
                         "free-tier roster entry is a --base-url/--model/--api-key-env "
                         "change, never a code change.")
    ap.add_argument("--base-url", default=None,
                    help="required for --provider openai_compatible (e.g. Google AI "
                         "Studio's or OpenRouter's OpenAI-compatible endpoint).")
    ap.add_argument("--api-key-env", default=None,
                    help="NAME of the env var holding the API key (never the key itself) "
                         "— required for --provider openai_compatible.")
    ap.add_argument("--rpm-limit", type=float, default=None,
                    help="proactive requests-per-minute ceiling (free tiers rate-limit "
                         "hard, e.g. 15 RPM) — paced BEFORE a 429, not just retried after.")
    ap.add_argument("--smoke", action="store_true",
                    help="ONE real call per arm to shake out a new provider/endpoint "
                         "before spending the real budget; writes no checkpoint.")
    ap.add_argument("--k", type=int, default=ve.RUN_K)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token-cost", type=float, default=0.0)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    repo_root = results_dir.resolve().parent if results_dir.name == "results" else Path.cwd()
    run_dir = _find_run_dir(results_dir, args.run)

    if args.provider == "openai_compatible":
        if not args.base_url or not args.api_key_env:
            ap.error("--provider openai_compatible requires --base-url and --api-key-env.")
        if not args.mock and not os.getenv(args.api_key_env):
            ap.error(f"Env var {args.api_key_env!r} is not set (use --mock for a wiring "
                     "smoke, or export the key first).")
    elif not args.mock and not os.getenv("ANTHROPIC_API_KEY"):
        ap.error("ANTHROPIC_API_KEY not set (use --mock for the wiring smoke).")

    if args.smoke:
        if args.mock:
            ap.error("--smoke makes REAL calls; it is meaningless with --mock.")
        run_smoke(run_dir, model=args.model, provider=args.provider, base_url=args.base_url,
                  api_key_env=args.api_key_env, rpm_limit=args.rpm_limit,
                  repo_root=repo_root, crop_cfg=CropConfig(), seed=args.seed,
                  folds=args.folds)
        return

    canonical = args.model == MODEL and args.provider == "anthropic"
    if not canonical:
        print(f"\n  *** NON-CANONICAL RUN: provider={args.provider} model={args.model} — "
              f"REHEARSAL / ARM-C. The locked Stage-3 headline is provider=anthropic "
              f"model={MODEL}; these results get their own artifact namespace and are "
              "NEVER written to summary.md. ***\n")

    try:
        res, comp, states_a, states_b = run_two_arm(
            run_dir, k=args.k, mock=args.mock, seed=args.seed, folds=args.folds,
            token_cost=args.token_cost, lambda_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            repo_root=repo_root, max_items=args.max_items, crop_cfg=CropConfig(),
            model=args.model, provider=args.provider, base_url=args.base_url,
            api_key_env=args.api_key_env, rpm_limit=args.rpm_limit)
    except SubstrateError as e:
        print(f"\n[SUBSTRATE GUARD] {e}\n")
        raise SystemExit(2)

    csv_path = write_results(run_dir, states_a, states_b, args.mock, args.model, args.provider)
    body = (f"_Model: {args.model} (provider: {args.provider})_\n\n"
           + "\n".join(summary_lines(res, comp)) + "\n")
    suffix = _model_suffix(args.model, args.provider)
    if args.mock:
        (run_dir / (MOCK_PREFIX + f"vlm_crop_summary{suffix}.md")).write_text(
            MOCK_BANNER + body)
        print("  [MOCK] two-arm wiring smoke — NOT evidence.")
    elif canonical:
        with open(run_dir / "summary.md", "a") as f:
            f.write("\n" + body)
    else:
        # Rehearsal/ARM-C runs write their own summary file — summary.md stays
        # canonical-only (the locked anthropic/claude-sonnet-4-6 headline).
        (run_dir / f"vlm_crop_summary{suffix}.md").write_text(
            f"# ⚠️ NON-CANONICAL — provider={args.provider} model={args.model}, NOT the "
            f"locked headline (provider=anthropic model={MODEL})\n\n" + body)
    print_console(res, comp)
    print(f"  artifacts -> {csv_path.name} in {run_dir}\n")


if __name__ == "__main__":
    main()
