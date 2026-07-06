# CLAUDE.md — AIQS-Agent project memory

> Read this first every session. It is the durable memory for this project: the
> north star, the hard constraints, the target architecture, the pinned stack,
> and the current phase. Keep the **Current phase** and **Decision log** sections
> updated as work progresses.

## North star (what we're building)

An agentic decision + reasoning layer that sits **ON TOP OF** an off-the-shelf
anomaly detector and makes its outputs production-trustworthy. Core thesis: the
value in industrial visual inspection is **NOT** in the detector (commodity,
solved) but in the **DECISION layer** — reducing false rejects (overkill) via
cost-aware, calibrated, **abstaining** decisions with auditable reasoning traces.
We are **AI-agent problem solvers, not computer-vision researchers**.

We optimize a **business cost function** and the **false-reject rate**, NOT
detection AUROC.

## Hard constraints (do not violate)

- **DO NOT** build or train a custom anomaly-detection architecture. Use the
  **Anomalib** library off the shelf.
- **Python only** for the core. Reproducible: pin versions, use `uv`, commit the
  lockfile (`uv.lock` is committed).
- **Minimal and verifiable.** No premature abstraction, no speculative generality.
- **Every phase must produce a MEASURABLE result**, not just "it runs."

## Target architecture (context — built incrementally, NOT all at once)

```
Detector (Anomalib)
  -> calibration
  -> adjudication agent (LangGraph) gathering verification signals
       (rules + a VLM second-look via the Anthropic/Claude API
        + later similar-case retrieval)
  -> cost-matrix decision policy: PASS / FAIL / ESCALATE(->human)
  -> later: root-cause agent, memory, conversational copilot, FastAPI
```

Observability via **Langfuse** once an LLM enters.

## Stack (pinned) & host notes

**Host:** Intel (x86_64) macOS 12.7.6, Core i5-5350U, **2 cores @ 1.8 GHz,
CPU-only** (no CUDA, and **no MPS** — MPS is Apple-Silicon only).

**Why the detector stack is pinned to the anomalib 1.2 line (important):**
Official PyTorch macOS wheels are **arm64-only from torch 2.3 onward**; the last
x86_64-macOS torch is **2.2.2**. The latest anomalib (2.5) needs a newer torch
with no x86_64-macOS wheel, so it **cannot be installed on this host**. anomalib
**1.2.0** installs cleanly against torch 2.2.2 and — critically — still ships
**AUPIMO** (added in anomalib 1.1), so we keep every Phase-0 metric.

Verified-installable, contemporaneous pin (Python **3.11.15**, via `uv`):

| package      | version | notes |
| ------------ | ------- | ----- |
| anomalib     | 1.2.0   | `[core,vlm]` extra; AUPIMO present |
| torch        | 2.2.2   | last x86_64-macOS wheel |
| torchvision  | 0.17.2  | pairs with torch 2.2.2 |
| numpy        | 1.26.4  | `<2` — torch 2.2.2 uses the NumPy 1.x ABI |
| lightning    | 2.4.0   | capped `<2.5`; anomalib 1.2 leaves it unbounded |
| torchmetrics | 1.4.3   | capped `<1.5` for the same reason |
| pandas       | 2.2.3   | capped `<2.3` (avoid pandas 3.0 breakage) |

**Upgrade path:** on an arm64 (Apple Silicon) or CUDA host, drop these caps and
move to **anomalib 2.x + current torch**. The Phase-1+ stack (LangGraph, VLM,
FastAPI, Langfuse) is independent of this detector pin.

**anomalib 1.2.0 packaging gotchas (already handled in pyproject):** the wheel
under-declares deps and eagerly imports *every* model at
`anomalib.models.__init__`. To make a bare `import anomalib.models` work we add:
`scikit-learn`, `requests`, and the `[vlm]` extra (`openai` / `ollama` /
`transformers` / `python-dotenv`) — none used by Phase 0, only needed so the
import succeeds. `ollama` is pinned `<0.4` because anomalib's VLM backend imports
`ollama._client._encode_image`, removed in newer ollama.

**Dataset access — anomalib's MVTec download URL is dead (HTTP 404).** anomalib
1.2.0 hard-codes an old mydrive.ch share that 404s. `src/aiqs/prepare_data.py`
(entry point `aiqs-prepare-data`, Make target `data`) fetches **only the
requested category** from the public HF mirror `TheoM55/mvtec_anomaly_detection`
and reorganises it into anomalib's native layout
(`<root>/<cat>/{train/good,test/<defect>,ground_truth/<defect>/<id>_mask.png}`).
anomalib then finds the folder and skips its dead download. `make train`/`smoke`
depend on `data`, so this is automatic.

## Repo conventions

- Package: `src/aiqs/` (src layout). Entry points: `aiqs-prepare-data`,
  `aiqs-train`, `aiqs-eval`.
- Config: `configs/*.yaml`, typed in `src/aiqs/config.py`; **category is
  configurable** and every field is CLI-overridable.
- Task runner: `Makefile` (`install` / `data` / `train` / `eval` / `baseline` /
  `smoke`).
- Results: `results/metrics.csv` (one row per eval) + `results/decisions.csv` (one
  row per `aiqs-decide`) + `results/runs/<id>/` (`summary.md`, `image_scores.csv`,
  Phase-1 `risk_coverage{,_target}.png` + `decision_scores.csv`) are **committed** so
  configs compare over time. Datasets, checkpoints, and heavy per-pixel dumps are
  gitignored (the risk-coverage plots are explicitly un-ignored).
- The eval backbone (`src/aiqs/eval/`) is the spine every later phase is measured
  against; `eval/decision.py` is the Phase-1 decision contract + calibration/policy
  (implemented). Entry points: `aiqs-decide` (real), `aiqs-sim-decision` (synthetic
  machinery validation, walled off under `results/synthetic_validation/`).

## Current phase

**Phase 0 — foundation + detection baseline + eval skeleton. ✅ COMPLETE.**
No LLM, no agent.

- [x] uv project (src layout, pyproject, committed lockfile), Makefile, README,
      .gitignore, `git init`.
- [x] Pinned, verified-installable Intel-mac detector stack.
- [x] Train EfficientAD on one MVTec AD category (default: `screw`).
  **SUPERSEDED**: default detector switched to **PatchCore** (see decision log
  2026-06-22 entry). EfficientAD configs preserved at `configs/default.yaml` +
  `configs/baseline_cpu.yaml` for reference.
- [x] Eval backbone: image AUROC, pixel AUPRO, AUPIMO (all working in 1.2.0).
- [x] Persist results + print baseline summary.

**Baseline (reduced budget, 600 steps, CPU, `configs/baseline_cpu.yaml`):**
pixel AUROC **0.940** · pixel AUPRO **0.821** · pixel AUPIMO 0.0023 · image AUROC
**0.559** · image F1 0.685. Strong localisation, weak image-level separation — the
untrustworthy-detector case the Phase-1 layer targets. Full 70k-step run deferred to
a GPU/arm64 host (`configs/default.yaml`). See `docs/PHASE0_REPORT.md`.

Default category: **`screw`** (small/subtle defects → many borderline scores →
good material for the Phase-1 adjudication layer). Fully configurable.

**Phase 1 — calibrated, cost-aware, abstaining decision layer. ✅ COMPLETE & VALIDATED
(layer + harness + honesty guard). Positive headline PENDING a real-separation
detector.** Deterministic spine only — NO LLM/agent yet (that is a later phase).

- [x] `eval/decision.py`: cost-matrix policy (`expected-cost argmin`, tie-break
      ESCALATE>FAIL>PASS), conformal calibration, metrics, guard — all pure
      numpy/sklearn (no torch).
- [x] **Calibration = cross (out-of-fold) Venn-Abers** (PRIMARY) + single 50/50 split
      inductive Venn-Abers (SECONDARY reference). See decision-log for why **not MAPIE**.
- [x] **Prevalence (label-shift) correction**: prior-shift of probabilities +
      importance-weighted metrics to a target production defect rate (default 2%),
      applied consistently to calibration and evaluation.
- [x] Risk-coverage sweep (escalation-cost knob 0→30/13≈2.31); BOTH risk axes
      (overkill rate + per-item cost); naive baseline = **cost-optimal** fixed
      threshold (+ calibration-free margin curve), matched-escape apples-to-apples.
- [x] Honesty guard: STOP on signal-free input (std≈0 / AUROC∈[0.47,0.53] / too few to
      K-fold); non-fatal WEAK/SMALL-n warnings otherwise.
- [x] `make decide RUN=<id>` (default latest), `make sim`, `make test`. **23/23 tests.**

**Two detectors run through the SAME decision layer — it reports an OPERATING ENVELOPE,
not a universal win:**

1. **600-step EfficientAD (image-AUROC 0.559) — HONEST NULL.** Guard refused a
   false-positive headline; calibrated `P(defective)` collapses to ≈ the base rate for
   both classes (good med 0.69 vs defective 0.72). Native → FAIL-all optimal; target-2%
   → PASS-all. No separable signal for any policy. (Weak-detector corner of the envelope.)
2. **PatchCore on Colab GPU (image-AUROC 0.976) — STRONG detector.** Now there IS signal:
   native (10/3/1) OURS cuts **overkill 0.293→0.160 and escape→0** via abstention, but at
   **review cost = 1** the escalation overhead exceeds the error savings, so the
   cost-optimal naive THRESHOLD wins on TOTAL cost (ours 0.2375 vs naive 0.2250). This is
   the **strong-detector / expensive-review** corner — NOT a null. (Colab run used the
   pre-fix decide; the **break-even review cost** + **realistic-matrix** numbers below come
   from re-running the updated `make decide` on that run's `image_scores.csv`.)

**Operating envelope (the real Phase-1 finding).** Cost-aware abstention beats a tuned
threshold when: **(a)** review is cheap (below the **break-even review cost**, reported
per run from the full anti-cherry-pick sweep in `breakeven.csv`); **(b)** the detector is
weak/uncertain (many borderline items); or **(c)** costs are **escape-dominant at low
prevalence**. The benchmark's illustrative **10/3/1** at 2% makes escapes too cheap →
PASS-all optimal (escape-rate→1.0, a cost-matrix property, NOT a bug); under a realistic
**100/3/1** (shipping a defect ≫ a re-inspection) the trade-off returns and OURS beats the
threshold. With a strong detector + expensive review + mild asymmetry, a tuned threshold
already suffices — the layer says **which regime you're in**. Synthetic check (`make sim`,
AUROC 0.92): OURS beats naive 21%/11% where separation exists.

**Phase 2A — VLM second-look on the ESCALATE bucket. 🟡 BACKBONE COMPLETE (mock-tested);
real-data headline PENDING a hard category + Colab anomaly-map export.** The FIRST LLM
component. Layers a vision-LLM adjudication step ON TOP of the Phase-1 spine, ESCALATE-only.

- [x] `src/aiqs/vlm/` — plain-function abstain pipeline around a node-shaped seam
      (`VLMState` + `adjudicate(state)->state`). **NO LangGraph in 2A** (E1, locked): a
      single-node/single-pass graph violates "no premature abstraction"; 2B wraps the seam
      as a node with no re-scaffold. Single model **claude-sonnet-4-6 vision**, NO Opus tier.
- [x] Calibrated abstain RULE = Phase-1 `decide_one` on a PROVISIONAL/UNCALIBRATED
      `p_vlm` (confidence→p map + shrinkage λ; full VLM calibration deferred to 2B). The
      100/3/1 matrix self-floors auto-PASS of a likely defect. Single-pass (VLM→human).
- [x] `src/aiqs/eval/vlm_eval.py` — the heart: (a) raw accuracy + confusion; (b)
      **error-INDEPENDENCE** vs the detector with the PINNED "detector wrong" = Phase-1
      naive cost-optimal threshold call, and the PRE-REGISTERED rule (load-bearing
      `Wilson-lo[P(VLM correct|det wrong)] > 0.50`; corroborating `κ<0.20` with bootstrap
      CI, never load-bearing alone); (c) bidirectional value (rescued→PASS / correct FAIL);
      (d) effective-review-cost BAND over λ vs Phase-1 break-even 0.868; (e) **K-run
      rule-stability** — rule recomputed each of K=5 runs, headline = modal + "x/5".
- [x] Substrate guard in code: STOP if ESCALATE∩good < 15, warn < 30.
- [x] `aiqs-vlm` / `make vlm RUN=<id> [MOCK=1]`; Langfuse wired (no-op without keys);
      anthropic+langfuse+pydantic deps resolved (lockfile, protobuf 7→6 harmless). **55/55
      tests (28 Phase-1 + 27 Phase-2A, all VLM mocked — no API).**
- [x] Wiring smoke on screw native-74 (bucket 26 = 16 good/10 def) passes end-to-end;
      mock output is WALLED OFF (`mock_vlm_*`, gitignored, loud banner — never touches the
      real `summary.md`), exactly like the synthetic validation. screw = smoke, NOT headline.

**Next (real 2A headline) — Phase-2B path.** Standard MVTec saturates (~0.976 image-AUROC
on BOTH screw AND capsule → ESCALATE bucket too small / `n_dw` underpowered; see 2026-06-29
log), so the headline moves to **MVTec AD 2** (reality-gap, detection genuinely hard):
anomalib 2.x as a GPU-host-only optional extra → train a hard AD2 category → **anomaly-map
export + crop** → `make decide` to verify ESCALATE∩good AND `n_dw` ≥ ~30 → two-arm
full-vs-crop VLM experiment (Stage 3). capsule/screw = wiring/substrate probes only, NOT
the headline.

**Phase 2B — hard-substrate hunt + the two-arm full-vs-crop experiment. 🟡 ENGINEERING
COMPLETE; HEADLINE EVIDENCE PENDING (needs a paid Kaggle-GPU session).** MVTec AD 2's
download is form-gated (not scriptable) — see decision log — so substrate hunting moved to
**VisA** (auto-downloads, no form).

- [x] **Stage 0** — pre-AD2/AD-migration hygiene: canonical-run provenance cleanup, live
      model-ID check (no silent downgrade), token-budget measured (<$0.50 single-pass).
- [x] **Stage 1** — version-dispatched detector/data seam (anomalib 1.2 local /
      2.x-GPU-host, zero regression to the pinned stack) + the anomaly-map **crop
      instrument** (`src/aiqs/crop.py`, detector-free, locally testable; DIFFUSE is a
      first-class outcome, not a crash).
- [x] **Stage 2** — VisA substrate sweep (Kaggle GPU): **THREE categories pass the
      pre-registered gate** (ESCALATE∩good AND n_dw ≥ 30): `capsules` (0.739 AUROC, 57/54),
      `macaroni1` (0.815, 83/45), `macaroni2` (rejected — 87% escalation ≈ VLM-on-everything).
      Ground = **capsules** (primary), `macaroni1` conditional on the capsules result.
- [x] **Stage 3** — two-arm full-vs-crop experiment, PRE-REGISTERED escape-classification
      rules (`vlm/reasoning_rules.py`, frozen before any real data), arm independence
      enforced in code (fresh single-turn calls, no shared conversation), checkpoint/resume
      (a crash loses ≤1 call), served-model stop guard, loud parse-failure fallback.
      **Crop-instrument dry-run #1 was VOID** (anomalib-2.x map normalization defeated the
      original diffuse test) — fixed and re-validated against real maps (19/20 crop, was
      1/20). **Haiku rehearsal COMPLETE** ($1.77, 1090/1090 calls, checkpoint/resume/parse-
      fallback all fired correctly in production): full-image Haiku is a rubber stamp
      (545/545 "clean"); crop fixes only 2% of escapes, **94% classify SEMANTIC**
      (pre-registered rules; labeling adequate). Confidence AUC 0.50, escapes 100%
      stable-wrong — both replicate the 2A observation that self-reported confidence and
      K-run agreement carry no independent signal.
- [x] **Degeneracy guard** (added to the shared `eval/vlm_eval.py` BEFORE the headline run,
      pre-registered `DEGENERATE_VERDICT_FRAC=0.95`): forces `"invalid-degenerate"` when one
      verdict covers ≥95% of a run, closing the exact spurious-"independent" failure mode the
      Haiku rehearsal exposed. Applies to every tier, including the not-yet-run headline —
      a validity precondition, not a change to the locked Sonnet design.
- [x] **ARM-C** — provider-agnostic `OpenAICompatibleVLMBackend` (the downward model-tier
      lever), reusing the same crop instrument/checkpoint/served-model-guard machinery;
      `aiqs-model-tier-report` for the cross-tier comparison table. Config-only roster swap
      (`configs/free_vlm_roster.example.yaml`, documentation-only).
- [ ] **PENDING — cannot be executed from this host:** the real claude-sonnet-4-6 headline
      run and any real ARM-C free-tier run. Both need the VisA images + anomaly maps, which
      exist only on the Kaggle GPU session that produced them (gitignored, not locally
      present) — the detector/value file-interface split that enables local development also
      means the real VLM call runs where the data lives. Ready-to-paste Kaggle cells handed
      to the user; **115/115 tests green**, every identified engineering risk closed.

## Decision log

- **2026-06-21** — Detector pinned to anomalib **1.2.0 / torch 2.2.2** (not latest
  2.5) because this Intel-mac host has no installable wheel for newer torch. Keeps
  AUPIMO. User approved the version fork.
- **2026-06-21** — Training runs **CPU-only** on a weak 2-core CPU; the paper-grade
  70k-step EfficientAD budget is many hours here. Workflow: `make smoke` to verify
  wiring, then a real (possibly reduced-step) baseline locally; code kept
  device-agnostic for a full run on stronger hardware later.
- **2026-06-22** — Ran the reduced-budget (600-step) CPU baseline (option A, user's
  choice). Measured ~8.4 s/step. Pre-downloaded real ImageNette (~1.5 GB) in parallel.
  Added `detector._silence_visualization()` (no-ops anomalib's viz callback) to cut the
  eval test pass ~13 min → ~4 min. Result recorded above + in the report.
- **2026-06-21** — anomalib's MVTec download URL 404s. Kept category **`screw`**
  (user's choice) by adding `prepare_data.py` to fetch screw-only from a public HF
  mirror and reorganise into anomalib's layout — works around the broken download
  without changing the approved architecture. (Turnkey alt was `capsule`, which
  has a clean single-tarball mirror; not needed since the screw adapter works.)
- **2026-06-22** — Phase-1 **calibration = Venn-Abers, implemented directly (NOT
  MAPIE)**. MAPIE's classification API emits prediction *sets*, but the expected-cost
  argmin needs a scalar `P(defective)`; and our "model" is a single 1-D score, not a
  fitted sklearn estimator with `predict_proba`. Venn-Abers (conformal family) yields
  exactly that calibrated scalar in ~30 lines on `sklearn.IsotonicRegression`
  (already shipped) — no new dependency, no fighting the library. **PRIMARY = cross /
  out-of-fold Venn-Abers** (every item predicted by a calibrator that never saw it →
  leakage-free, uses all data; the right call given only **41 normal parts** make a
  single 50/50 split too fragile). SECONDARY = single-split inductive Venn-Abers as a
  spec-literal reference. Merge `p = p1/(1-p0+p1)`. User-approved.
- **2026-06-22** — **Cost matrix LOCKED (relative):** false-accept/escape **10**,
  false-reject/overkill **3**, escalate/review **1**, correct PASS/FAIL **0**. At
  these costs the optimal policy is PASS `p≤0.10` / ESCALATE `0.10<p<0.667` / FAIL
  `p≥0.667`; ties break ESCALATE>FAIL>PASS (never silently PASS uncertain).
- **2026-06-22** — **Phase-1 result on the 600-step detector is an HONEST NULL** (see
  Current phase). Decided (user) to **commit it honestly**: the guard refusing a
  false-positive headline on a 0.559-AUROC detector is the deliverable's integrity
  feature. Kept the null plots; wrote up the root cause (weak detector + 74% base rate
  ⇒ FAIL-all/PASS-all near-optimal). Positive headline deferred to a GPU baseline.
- **2026-06-22** — Added **prevalence (label-shift) reweighting** (REQUIRED, user): the
  74%-defective benchmark inverts production economics. Under the label-shift
  assumption (class-conditional score densities invariant), `prior_shift` corrects the
  Venn-Abers probabilities to a target prior (Saerens/Elkan odds rescale) and
  `prevalence_weights` importance-weights the metrics — same `(target/source)` ratios,
  so calibration and evaluation stay consistent. We **weight, not subsample** (only 41
  normal parts). Default target prevalence **2%**.
- **2026-06-22** — Added a **LABELED synthetic machinery validation** (`aiqs-sim-decision`,
  option B), walled off under `results/synthetic_validation/` with a loud SYNTHETIC
  banner and never written to `results/decisions.csv`. Purpose: prove the code is
  correct on a separating detector, NOT that the approach works on real data.
- **2026-06-22** — Dev tooling: **pytest** pinned in a `[dependency-groups] dev` group
  (`>=8,<9`) — test-only, never enters the runtime resolve. `make test` /
  `uv run --group dev pytest`.
- **2026-06-22** — **Detector switched to PatchCore** (from EfficientAD). Rationale:
  PatchCore builds a coreset memory-bank in a **single epoch** (no 70k-step training
  grind) and is typically **stronger at image-level separation** — exactly what the
  Phase-1 decision layer needs. EfficientAD optimises inference speed, which is
  irrelevant for the demo/eval pipeline. The harness was already detector-configurable
  (`detector.py` + `config.py`); changes: new `configs/patchcore_cpu.yaml` (default),
  `build_train_engine` now branches step-driven vs epoch-driven, Makefile default
  flipped. EfficientAD configs retained.
- **2026-06-22** — **PatchCore runs on Colab GPU, NOT locally.** WRN50-2 + coreset 0.1
  on this 2-core/limited-RAM Intel-mac **stalled ~3h with no output** (RAM thrash /
  greedy-coreset bound). Lowered `configs/patchcore_cpu.yaml` to **coreset 0.01 + batch
  8** (PatchCore paper: 1% coreset ≈ full accuracy) for any local attempt, but the real
  baseline is Colab (image-AUROC **0.976**). Do NOT re-run the detector locally to
  "cross-check" a Colab score — different stack (anomalib 1.2/torch 2.2/CPU vs 2.x/CUDA)
  → scores differ legitimately. The meaningful local check is `aiqs-decide` on the Colab
  `image_scores.csv` (detector-free, deterministic). Also fixed `_load_weights` to
  `resize_` PatchCore's empty `memory_bank` buffer before load (eval was untested).
- **2026-06-22** — **Phase-1 reframed as an OPERATING-ENVELOPE result** (user, on the
  0.976 PatchCore run). The old auto-verdict said "no separation to exploit" whenever
  OURS cost ≥ naive — **wrong** for a strong detector. Fixed `_verdict` to distinguish
  (a) weak-detector null from (b) strong-detector / **expensive-review** (abstention
  cuts overkill+escape but review overhead > savings at review=1). Added: **break-even
  review-cost sweep** (`breakeven_review_cost`; the swept value IS the true review cost,
  driving policy AND accounting — distinct from the abstention-aggressiveness sweep; full
  curve in `breakeven.csv`/`risk_coverage_breakeven.png`; sanity: review→0 MUST beat
  naive); a **realistic escape-dominant matrix 100/3/1** for the low-prevalence regime
  (the illustrative 10/3/1 makes escapes too cheap at 2% → PASS-all, a cost-matrix
  property not a bug) reported side-by-side with 10/3/1; and an adaptive operating-
  envelope writeup. Do NOT change the LOCKED matrix to manufacture a win — the full sweep
  + domain-justified realistic matrix are the anti-cherry-pick controls.
- **2026-06-23** — **Phase-2A substrate finding (zeroth step before any LLM code).**
  Counted the ESCALATE bucket on the 0.976 PatchCore run, split by label: native-74%
  ESCALATE∩good=**16**; target-2% ESCALATE∩good=**1 (DEAD)**. Prior-shift to 2% drops
  defect odds ~140× and a strong detector collapses good parts to p≈0 (PASS), EMPTYING the
  borderline band of goods (reweighting does NOT fatten it — the user's hypothesis, falsified
  by data). ⇒ standard MVTec `screw` cannot show the overkill-reduction lever at thesis
  prevalence; the lever needs a genuinely HARD category/AD2. Operating-envelope BOUNDARY,
  not a failure. screw = wiring smoke only.
- **2026-06-23** — **Phase-2A design LOCKED, then built (backbone).** ESCALATE-only; single
  claude-sonnet-4-6 (no Opus tier → 2B); calibrated abstain RULE on a PROVISIONAL/UNCALIBRATED
  p_vlm (Venn-Abers VLM calibration → 2B) reported as a shrinkage-λ SENSITIVITY BAND led from
  the conservative end; **E1: no LangGraph in 2A** (single-node/single-pass violates "no
  premature abstraction" — built as plain functions around a `VLMState`+`adjudicate` seam
  that 2B wraps as a node); full-image-only now (a SIGNAL-PATH change, not just small-n — the
  crop run is the real mechanism); **"detector wrong" PINNED** to the Phase-1 naive
  cost-optimal threshold (honest no-layer comparator); **PRE-REGISTERED error-independence
  rule** (load-bearing Wilson-lo[P(VLM correct|det wrong)]>0.50; κ<0.20 corroborating-only
  with bootstrap CI; recomputed per K-run → modal + K-stability headline). Deps
  anthropic/langfuse/pydantic resolve on the pinned stack (desk-check + uv-resolve GREEN;
  protobuf 7→6 harmless). Mock smoke walled off (`mock_vlm_*`, gitignored). 55/55 tests.
  User ran a propose→confirm loop on every choice (incl. a Dispatch review that sharpened
  the pinned "detector wrong" def and the CI-guarded/κ-secondary rule). Implemented locally
  (VLM = API call, not GPU); real headline still needs a hard category + Colab map export.
- **2026-06-23** — **`evaluate._load_weights` made tensor-safe for PatchCore memory-bank
  checkpoints.** The shape-diagnostic comprehension assumed every state_dict value was a
  tensor and called `.shape`; PatchCore checkpoints carry non-tensor entries (strings/
  scalars/metadata), so it crashed (`'str' object has no attribute 'shape'`) on every
  non-screw category on Kaggle/Colab. Fix: guard the shape pass with
  `isinstance(v, torch.Tensor)` (detector-agnostic — no model-name special-casing).
  `strict=False` retained for the missing eval-metric buffers; the buffer-resize logic is
  unchanged. Added: `_load_weights` now RETURNS the load result and WARNS (RuntimeWarning)
  when any registered model buffer lands in `missing_keys` — guards against strict=False
  silently swallowing an unrestored `memory_bank` (→ degenerate ~0.5 AUROC). Regression
  test (`tests/test_evaluate_load.py`): a state_dict with non-tensor values does not raise
  and tensors load; an omitted buffer surfaces in missing_keys + warns. 57/57 tests.
- **2026-06-29** — **Phase-2B Stage-0.3 provenance hygiene (pre-AD2 cleanup, no API/Colab).**
  Audited the committed results state before the AD2 migration; found it dirtier than the
  roadmap assumed and fixed it locally. (1) `results/decisions.csv` carried a leftover git
  **merge marker** (`<<<<<<< HEAD` fused into the header) + 7 junk rows + a phantom leading
  column — committed and propagated across several commits (predates f0ef3d6). The writer is
  `pd.DataFrame([row]).to_csv(index=False)` (21 cols), so the next `aiqs-decide` would have
  mangled it further. Rebuilt clean: 21-col header, deduped to the two authoritative decide
  rows (screw 130823Z + canonical capsule 142659Z); pandas round-trip verified (2×21, no
  `Unnamed` column). (2) **Three near-identical capsule run dirs** (140858Z full, 141717Z
  partial/stray, 142659Z full) with **identical `image_scores`**, and 142659Z re-decided 5×
  → 5 redundant `decisions.csv` rows. Picked **142659Z as canonical** (the dir the VLM was
  attempted on); `git rm`'d 140858Z + 141717Z. (3) **The 2A VLM capsule "headline" is NOT a
  committed artifact**: 142659Z `vlm_results.csv` was **1 byte** (a lone newline); the only
  real VLM output in-repo is the WALLED-OFF screw mock (`mock_vlm_*`, gitignored — the wall
  holds). The cited capsule VLM numbers (16/19 rescue, 2/5 escape) came from the Colab fork
  **discarded in 2ebbdeb** → NOT authoritative. Removed the empty stub. Capsule's role is now
  honestly **substrate-probe only** (ESCALATE∩good=19, n_dw=5 → underpowered independence
  test); the real VLM headline is **PENDING on a substrate-verified AD2 category** (Stage 3).
  Stage-0.1 (live model-ID log) + 0.2 (token budget from Langfuse) remain TODO — both need an
  API key / Langfuse trace **absent on this local host**, so they belong to the planned Kaggle
  session, not this cleanup. No code touched; 57/57 tests unaffected.
- **2026-06-29** — **Phase-2B Stage 0.1 + 0.2 DONE — live model-ID + token budget, run
  LOCALLY (no Kaggle).** The VLM is an API call, not GPU (per the 2A note), so this ran on
  the Intel-mac host with an `ANTHROPIC_API_KEY` in a local `.env` (gitignored, commit
  `ee2e8d4`). New `scripts/verify_vlm_local.py` — a pre-flight that REUSES the production
  `AnthropicVLMBackend` content construction (no fork) and makes 8 REAL `claude-sonnet-4-6`
  calls on local screw images.
  - **0.1 GREEN.** All 8 calls served `claude-sonnet-4-6`, matching the `MODEL` const in
    `vlm/backend.py`. **No silent 3.5-downgrade** — the fork's failure mode is cleared. The
    script STOPS LOUD (exit 2) if the served model ≠ expected (pre-registered guard, not a
    silent work-around).
  - **0.2.** Per call ≈ **639 input** (constant — fixed 512px image + fixed prompt) +
    **~112 output** (95–129) ≈ **751 tokens**. At claude-sonnet-4-6 pricing ($3/$15 per 1M
    in/out): per call ≈ **$0.0036**; 8-call pre-flight ≈ **$0.03**. AD2 full-image
    projection: bucket=40 ≈ **$0.14**, bucket=80 ≈ **$0.29**; crop adds a 2nd image block
    (+~500 img-tokens/call) → a powered two-arm run stays **< ~$0.50**. **Token budget is
    NOT a constraint on the AD2 run.** Measured on screw (capsule images are Colab-only
    here); per-call cost is image-size + prompt driven → category-independent, transfers as
    a representative estimate. Crop's exact increment is measured in Stage 1 once the crop
    instrument exists.
  - **Stage 0 (0.1 + 0.2 + 0.3) COMPLETE.** Hygiene gates green → ready for Prompt A
    (Stage 1: anomalib 2.x GPU-host-only optional extra + MVTec AD 2 datamodule +
    anomaly-map crop), pending user go-ahead and a propose→confirm outline.
- **2026-06-29** — **Phase-2B Stage 1 — version-dispatch backend + MVTec AD 2 datamodule +
  anomaly-map CROP INSTRUMENT.** Pure-python core LOCAL-VERIFIED; the anomalib-2.x backend is
  GPU-host-pending (the Intel-mac cannot install 2.x). User approved (propose→confirm) all
  three design choices: isolated `_backend_v2` via version dispatch; native `MVTecAD2` +
  `test_public` (GT) split; core-local / 2.x-GPU-marked split.
  - **Version dispatch (zero local regression).** `_anomalib_compat.anomalib_major()`;
    `detector.py`/`data.py` are version-safe seams — NO top-level anomalib import (that crashes
    under 2.x, where the 1.2 `MVTec` class was removed); they lazily run the 1.2 bodies in-line
    or delegate to `_detector_v2.py`/`_data_v2.py` when `anomalib.__version__` >= 2. Locally the
    seam still resolves to v1 (66/66 tests; smoke unaffected).
  - **MVTec AD 2 (verified 2.x API).** `_data_v2` builds `MVTecAD` (the `MVTec` rename) or
    `MVTecAD2`; GT comes from the **public-test** split offline (`test_type`) — private/mixed
    need the eval server, out of scope for our labelled metrics. `_detector_v2` uses the 2.x
    Engine (no `task=`/`image_metrics=`; metrics via Evaluator/sklearn) and `engine.predict()`
    -> `ImageBatch` (`.pred_score` / `.gt_label` / `.image_path` / `.anomaly_map`).
  - **CROP INSTRUMENT (core, detector-free, LOCAL-VERIFIED).** `aiqs/crop.py`
    `compute_crop(map, image, cfg)` — RELATIVE-TO-MAX peak threshold (a percentile collapses on
    sparse peaks -> selects the whole frame; learned via a failing test) -> padded, min-size,
    clamped bbox -> high-res crop. **DIFFUSE is first-class** (flat map by peak/mean, OR peak
    area > frac -> `crop=None`, full-image fallback — a measured signal for the Stage-3
    perception-vs-semantic split, not a crash). `aiqs/vlm/crop_fn.make_crop_fn(cfg)` plugs into
    the EXISTING 2A backend seam (`crop_fn` / `anomaly_map_path`); `cfg.crop.enabled` is the
    ARM-A(full) / ARM-B(full+crop) toggle, and a diffuse item is byte-identical to ARM-A. Map
    export: `evaluate._write_anomaly_maps` dumps per-image `.npy` (gitignored) + a manifest. 9
    new crop tests (peak / diffuse / min-size / border / crop_fn) -> **66/66**.
  - **DEP FINDING (surfaced, not hacked).** A co-locked `uv` `ad2` extra is INFEASIBLE: the
    base caps (`torchmetrics<1.5`, `numpy<2`, `lightning<2.5`, `torch==2.2.2`) are mutually
    exclusive with anomalib >= 2.2 (needs `torchmetrics>=1.8.2`, numpy 2.x, lightning 2.6) —
    MEASURED with `uv pip compile` (`No solution found ... torchmetrics`). Delivered the GPU-host
    stack as a SEPARATE `requirements-ad2.txt` (fresh env + `pip install -e . --no-deps`). The
    cleaner fix — split the detector stack into conflicting `det1`/`ad2` extras with a
    detector-free base (matches the thesis) — is **deferred to a user decision** (base refactor).
  - **Pending on the GPU host (not run locally).** Install `requirements-ad2.txt` -> confirm the
    exact `MVTecAD2(test_type=...)` value + auto-download vs manual; train an AD2 category; run
    `aiqs-eval` -> image_scores.csv + anomaly maps; then `make decide` for the Stage-2 substrate
    count (ESCALATE∩good AND n_dw >= ~30).
- **2026-06-29** — **Det-stack split DEFERRED until after the GPU-host AD2 round (user decision).**
  Rationale (cause->effect): the `image_scores.csv` + anomaly-map FILE interface already gives the
  decoupling the experiment needs — the detector world (anomalib 2.x, GPU) and the value world
  (pinned stack, local) talk by FILE, not import (the Phase-0 design). The torch-free-base refactor
  would FORMALISE that decoupling in the dependency graph but yields ZERO new capability for Stages
  2-3; it also risks regressing the working base (66/66) right before the critical experiment for
  no evidence-gain, and its value is contingent on AD2 being the right path (if Stage 2 yields no
  substrate or Stage 3 is null, the polished 2.x detector path may be unneeded). The Stage-1
  dep-conflict IS the proof the refactor is the correct END-STATE (the value-layer caps clash with
  anomalib 2.x because the two still share one dependency world) — but the medicine is taken AFTER
  the experiment: a capstone if positive, or a necessity if the two-stack workaround creates real
  GPU-host friction. **Two tactical guards for the GPU round** (the `_data_v2`/`_detector_v2`
  modules are WRITTEN BUT NEVER RUN — 2.x API match unverified): (1) **SMOKE FIRST** — one tiny run
  to shake out the 2.x API (`MVTecAD2` ctor args, PatchCore 2.x args, `predict()`->`ImageBatch`
  `.anomaly_map` export) BEFORE any full train; expect 1-2 API mismatches, catch them in a 5-min
  smoke not a 40-min train (same discipline as the capsule VLM smoke). (2) **SUBSTRATE GATE =
  image-level uncertainty, NOT "it's AD2"** — AD2's difficulty is PIXEL-level; PatchCore can give
  low pixel-AUPRO but high image-AUROC -> empty ESCALATE bucket again (standard-MVTec repeat). Read
  **image-AUROC FIRST** (~0.97 => no substrate; lower => candidate), THEN ESCALATE∩good AND n_dw.
- **2026-06-29** — **GPU-round de-risked locally: 2.x API SOURCE-verified + a paste-and-run runner.**
  Turned guard (1) — "smoke first, expect 1-2 API errors" — into a DOCUMENT shake-out: verified
  `_data_v2`/`_detector_v2` against the anomalib 2.x SOURCE (mvtecad2.py, patchcore lightning_model.py,
  Engine). They MATCH: `MVTecAD2(augmentations=..., test_type="public" [the default], AUTO-DOWNLOADS)`;
  `Patchcore(backbone/layers/coreset_sampling_ratio/num_neighbors)` (no `task`); `Engine(**kwargs ->
  Lightning Trainer)`; `predict() -> ImageBatch`. So no AD2 `prepare_data` parallel is needed, and the
  expected smoke breakage is now low. Added: `_detector_v2.smoke(cfg)` (1 train + 1 predict batch;
  asserts `ImageBatch.pred_score/image_path/anomaly_map` — fail in seconds, not 40 min);
  `configs/patchcore_ad2.yaml` (AD2 category list, crop off for the substrate round);
  `scripts/run_ad2_gpu.py` (REFUSES anomalib<2; `--smoke` then full train->eval->decide->`aiqs-vlm
  --mock`, surfacing the 3 numbers image_auroc / ESCALATE∩good / n_dw; `aiqs-vlm` rc==2 SubstrateError
  is a VALID "no substrate" outcome). Local-verified what is verifiable here (config parse, runner
  import under 1.2, py_compile, 66/66 hold); the 2.x run itself remains the user's GPU session.
- **2026-06-29** — **Substrate-hunt dataset: VisA, NOT MVTec AD 2 (user decision).** The first GPU
  smoke PROVED the 2.x backend is API-correct — anomalib 2.x import, `MVTecAD2` ctor, backbone
  download, `Engine.fit`, `prepare_data` all ran; it died ONLY on anomalib's dead AD2 download URL
  (a `mydrive.ch` 404 — the EXACT Phase-0 MVTec failure). Worse, AD2 has NO public HF/Kaggle mirror
  and the official download is FORM-gated (CC BY-NC-SA registration) -> not scriptable. Switched to
  **VisA**: anomalib's `Visa` datamodule AUTO-DOWNLOADS from a LIVE public S3 link
  (`amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`, verified in source), no
  form, and its test split carries GT offline (no eval server) -> sweep its 12 categories (candle,
  capsules, cashew, chewinggum, fryum, macaroni1/2, pcb1-4, pipe_fryum) cheaply for image-level
  substrate. **Same substrate gate** (image-AUROC first; the pixel-vs-image trap applies to VisA
  too). AD2 stays the high-fidelity fallback IF VisA yields nothing and the form/upload cost is
  accepted. Added `_data_v2._build_visa`, `configs/patchcore_visa.yaml`; made
  `scripts/run_ad2_gpu.py` config-driven (VisA default, `--category` optional override).
- **2026-07-02** — **Phase-2B Stage 2 GATE PASSED — VisA substrate sweep (Kaggle GPU, commit
  `3417b20`).** First the 2.x path validated END-TO-END on real hardware: the VisA smoke ran with
  ZERO API errors (source-verification held), then candle full round reproduced across two
  sessions (39/28/11). The user moved GPU work Colab→Kaggle (Colab VM recycling); sweep of 6
  categories pushed `image_scores.csv` per run, table recomputed LOCALLY (detector-free, own
  calibration + pinned "detector wrong"):
  | category | image-AUROC | bucket | ESC∩good | n_dw |
  |---|---|---|---|---|
  | candle | 0.972 | 39 | 28 | 11 |
  | **capsules** | **0.739** | **109** | **57** | **54** |
  | **macaroni1** | **0.815** | **138** | **83** | **45** |
  | macaroni2 | 0.646 | 174 | 92 | 80 |
  | pcb1 | 0.936 | 65 | 35 | 26 |
  | pcb2 | 0.928 | 74 | 49 | 29 |
  **THREE categories pass the pre-registered gate (ESC∩good AND n_dw >= 30): capsules, macaroni1,
  macaroni2.** The pixel-vs-image trap did NOT materialise on VisA — image-level detection is
  genuinely hard here (the reality-gap substrate AD2 was meant to provide, obtained without the
  form-gated download). candle's Phase-1 decide was also the first REAL positive for the decision
  layer (native 11% / realistic 13% cheaper than the tuned threshold). Bonus finding: two
  sessions, two hosts (Colab/Kaggle), same seed -> identical candle numbers (reproducibility).
  Stage-3 ground selection + Prompt-B design outline are the next propose→confirm.
- **2026-07-02** — **Phase-2B Stage 3 BUILT (mock-verified, 78/78) — two-arm full-vs-crop
  experiment + PRE-REGISTERED escape-classification rules, committed BEFORE the real run.**
  User approved ground=**capsules** (0.739, 57/54 — "weak but signalful", the thesis regime;
  macaroni2 rejected: 174/200 escalated ≈ VLM-on-everything, hollows out ESCALATE-only) with
  macaroni1 as the CONDITIONAL second ground, plus three confirm-conditions, all implemented:
  1. **PRE-REGISTERED labeling rules** (`vlm/reasoning_rules.py`, frozen 2026-07-02 pre-run —
     the anti-p-hacking condition): classification unit = (run,item) ARM-A escape, judged by
     its ARM-B pair. Priority: (i) **PERCEPTION** := ARM-B verdict flips to "defect" (objective,
     no text matching); (ii) **SEMANTIC** := ARM-B stays "clean" AND reasoning matches a frozen
     NORMALIZING regex family (artifact attributions: reflection/glare/lighting/illumination/
     shadow/dust/debris/smudge/artifact; "normal variation|appearance|surface|texture|feature";
     "within normal|acceptable|tolerance"; "acceptable|expected|typical + variation|appearance|
     feature|surface"; cosmetic; harmless — word-boundary guarded, "abnormal" does NOT match);
     (iii) **UNCLASSIFIED** := everything else, own rate reported; >0.30 of eligible escapes =>
     pre-registered verdict "rule-based labeling insufficient, human read required" — rules are
     NOT widened post hoc. DIFFUSE items (no crop; ARM-B byte-identical to A) are excluded from
     the denominator and counted separately.
  2. **Arm independence in code**: every call is a fresh single-turn `messages.create` (no
     shared conversation -> no anchoring); per-arm fresh state objects, same item order, same
     per-run seed schedule (tested). Plus a **served-model STOP guard** now in the backend
     itself: any call served != claude-sonnet-4-6 raises (the silent-downgrade lesson), and
     usage tokens are captured per call (the crop's 2nd-image increment is measured, not
     estimated).
  3. **macaroni1 TRIGGER**: run macaroni1 ONLY after the capsules result — positive => second
     envelope point (validation); null => different-regime test ("regime or VLM?"). Never in
     parallel (2x budget, and designing exp-2 before learning from exp-1). Code is
     ground-parametric (`--run`), runs are single-ground.
  Also: **ARM-A doubles as a free 2A replication** on new ground — reported as a descriptive
  line (good-rescue rate, defect-escape rate, confidence-separation AUC ~0.5 = does not
  separate); the discarded-fork numbers are never quoted as baseline. New: `eval/crop_eval.py`
  (paired escape comparison w/ fixed-by-crop|broken-by-crop discordants, stable-vs-flip
  [stable := escaped in ALL K runs — invisible to K-agreement], token-cost line),
  `vlm_crop.py` (`aiqs-vlm-crop`, `make vlm-crop`), 12 new tests. Honest budget correction
  surfaced pre-approval: bucket=109 x 2 arms x K=5 ≈ 1090 calls ≈ **$4-5** (the Stage-0
  "<$0.50" was single-pass). Real run happens on Kaggle (maps live there; entry point + guards
  are in-repo — no fork).
- **2026-07-02** — **Stage-3 dry-run #1 VOID (crop never engaged) -> crop instrument FIXED against
  real maps.** The first real capsules two-arm run ($~5, commit `06c762f` record) came back
  escape A 0.515 = B 0.515, **134/134 escapes diffuse-excluded** — the crop_fn produced a crop for
  ZERO items, so ARM-B was byte-identical to ARM-A: an INSTRUMENT failure, not a null result (the
  hypothesis was never tested). Diagnosis on a 20-map sample pushed from Kaggle
  (`capsules_maps_sample.tgz`): anomalib-2.x post-processor maps are normalized with a HIGH FLOOR
  (min~0, max~0.4, mean~0.25) -> the raw `peak/mean < 2.0` flat-map guard declared 19/20 maps
  diffuse. My "zero-background + sharp peak" assumption was synthetic-test-shaped, not real-map
  -shaped. FIX (instrument repair, NOT outcome tuning — the frozen pre-registered artifacts are
  the reasoning rules + Wilson threshold, and no VLM output feeds this): threshold = max(top-1%
  quantile of the normalized map [adapts to high-floor maps], `peak_fraction` rel-to-max floor
  [guards sparse maps]); diffuse = GEOMETRIC — the selected region's raw BBOX > `diffuse_area_frac`
  of the frame (scattered/plateau peaks -> crop ≈ full image ≈ no look-closer value). Dropped
  `diffuse_peak_to_mean` (meaningless on normalized maps); added `peak_top_frac=0.01`. Validated
  EMPIRICALLY on the 20 real maps: **19/20 crop (mostly focal bboxes), 1/20 genuinely diffuse**
  (was 1/20 crop). 78/78 tests. The $5 lesson institutionalised: dry-run the instrument against
  REAL exported maps before spending API budget — a mock smoke validates wiring, not thresholds.
- **2026-07-05** — **Stage-3 HAIKU REHEARSAL COMPLETE (Kaggle, $1.77, commit `71910cc`) — pipeline
  fully validated end-to-end with real money; a striking model-tier finding on the side.**
  Fresh capsules detector round (4 identical dup dirs — user re-ran the cell; newest 181237Z is
  the ground) -> two-arm run with `--model claude-haiku-4-5` (REHEARSAL namespace; the locked
  headline model claude-sonnet-4-6 is untouched, summary.md untouched).
  - **Engineering: every protection layer fired correctly in production.** 1090/1090 calls
    checkpointed; 5 parse failures fell back LOUD to unsure without crashing the run (the
    deterministic-malformed scenario is real — it happened); crop payload confirmed in-band
    (tokens/call A 809 -> B 1353, the +544 second image block); diffuse 2/109 (matches the mock
    gate); cost $1.77 vs the ~$1.7 estimate; model-namespaced artifacts kept the canonical run
    clean. The $5-loss failure class is closed.
  - **Haiku result (REHEARSAL-grade evidence, NOT the locked headline):** full-image Haiku is a
    RUBBER STAMP — verdict "clean" on 545/545 ARM-A calls (rescue 1.00 AND escape 1.00; accuracy
    = the bucket's good-rate). Crop barely moves it: escape 1.000 -> 0.962; of 250 eligible
    A-escapes, **perception=5 (2%), SEMANTIC=235 (94%)**, unclassified=10 (incl. the 5 parse
    fallbacks; rate 0.04 => labeling ADEQUATE — the pre-registered rules worked). Escapes are
    100% stable-wrong (52/52) -> K-run agreement is NOT an abstain signal here; confidence AUC
    0.50 -> not a signal either (2A observation replicated). The formal independence rule says
    "independent 5/5" in BOTH arms — but that is the all-clean bias being right exactly where
    the detector over-rejects (the rescue side); with escape=1.0 it is value-asymmetric, not a
    usable verdict. **For the haiku tier, the crop/perception hypothesis is REFUTED: the failure
    is semantic** (it sees the flagged region and calls it normal variation/artifact).
  - **Program position:** the sonnet-4-6 headline run is now maximally de-risked (~$5, one
    command, resume-safe). Rehearsal prediction to test: if sonnet's escapes are ALSO
    semantic-dominated, the 2B lever is prompt/anchor design, not better pixels.
- **2026-07-06** — **Degeneracy guard added to the SHARED `eval/vlm_eval.py`, BEFORE the
  sonnet-4-6 headline run — a pre-registered validity check, not a Sonnet-design change.**
  User directive: the guard exists because ANY tier (including the not-yet-run headline)
  could produce a rubber-stamp verdict distribution that satisfies `Wilson-lo>P_IND_MIN` by
  sheer luck of being "right" on whichever side the detector over-rejects — exactly what the
  Haiku rehearsal exposed (545/545 "clean", formally "independent 5/5", substantively
  meaningless). Applying the guard ONLY to non-canonical tiers would mean "the guard exists
  but doesn't apply to the manifest it's meant to protect" — a partially-applied validity
  guard is not a guard. This is read as a validity PRECONDITION established before the
  headline run, not an exception to "don't touch the locked Sonnet design" (cost matrix,
  classification rules, prompts, thresholds — all untouched).
  - `DEGENERATE_VERDICT_FRAC = 0.95` (pre-registered, frozen before any headline data
    exists): a run where one raw verdict (`defect`/`clean`/`unsure`) covers >=95% of calls
    is forced to `RuleOutcome.label = "invalid-degenerate"` regardless of the computed
    Wilson-lo/kappa. Operates on the RAW 3-way verdict, not the binary detector-comparison
    call (an all-"unsure" run is degenerate too, even though `vlm_call_label`'s arbitrary
    p_vlm-lean would hide that in the binary vector).
  - **Bugfix found while auditing consumers (per user's explicit ask):** `evaluate()`'s
    modal/stability aggregation used `max(dist, key=dist.get)` over a fixed 3-label dist —
    if ALL K runs were invalid-degenerate (dist all zero for independent/redundant/theatre),
    `max()`'s tie-break-by-insertion-order silently returned `"independent"`, printing a
    nonsensical `"YES: independent in 0/5 runs"`. Fixed: `rule_stability`'s YES/NO now gates
    on `n_indep > 0 AND n_indep == max(dist.values())` — a real, load-bearing correctness fix
    that predates and is independent of the new label (it would have misfired on the OLD
    3-label dist too, just needed the right degenerate-tie input to trigger).
  - Checked every downstream consumer of `RuleOutcome.label`/`rule_distribution` (the ask):
    `.is_independent` already handles any unknown label correctly (`== "independent"`);
    `vlm_decide.py`'s two print sites are plain dict/string interpolation — the new label
    surfaces automatically, no crash, no silent drop.
  - `error_independence()` gained an optional `raw_verdicts` kwarg (default `None` — fully
    backward compatible with the two existing unit tests that omit it); the one production
    call site in `evaluate()` now passes `[s.vlm_verdict for s in states]`.
  - 6 new tests (thresholds, forced-degenerate override, backward-compat, the tie-break
    bugfix itself). 101/101 at this point in the session.
- **2026-07-06** — **ARM-C: provider-agnostic OpenAI-compatible VLM backend — the downward
  model-tier lever ($0 free tier -> frontier), Sonnet headline design untouched.** Same
  Intelligent-APIs cost-scaling question the Haiku rehearsal opened: sweep model tier on the
  IDENTICAL bucket/rules/matrix and find where the mechanism breaks.
  - `vlm/backend_openai_compatible.py` (`OpenAICompatibleVLMBackend`): same call contract
    (`backend(state) -> VLMVerdict`), same `VLMVerdict`/`parse_verdict` (no fork), same
    served-model STOP guard (via the shared `model_guard` module, below), same crop
    instrument (`vlm.crop_fn.make_crop_fn`, just with an OpenAI-shaped `encode_fn` — the crop
    LOGIC never changes per provider). `base_url`/`model`/`api_key_env` are ALL
    caller-supplied — swapping a free-tier roster entry (they rotate monthly) is a config
    change, never a code change. **The API key is only ever taken as an ENV VAR NAME, never
    a literal** — a YAML/CLI config can never carry a secret (the key-hygiene lesson).
  - **Fork-prevention refactor (both backends now share):** `vlm/image_encode.py` (resize +
    PNG-b64 — was duplicated 2x, now 3x-would-have-been) and `vlm/model_guard.py`
    (`model_matches`/`assert_model_matches` — the "alias vs dated full id" logic). Fix the
    resize edge case once, both backends get it; no silent drift between copies.
  - **Rate limiting — two distinct mechanisms, not one:** a PROACTIVE `rpm_limit` sleep
    (paces BEFORE hitting a free tier's RPM ceiling, so the ~500/day quota isn't burned on
    429s) is separate from `max_retries` (the SDK's own exponential backoff AFTER a
    transient error — same pattern as the Anthropic backend's `max_retries=8` fix).
  - **Resume across day-boundaries is the EXISTING checkpoint/resume mechanism, not a new
    scheduler** — a free-tier daily cap means an ARM-C run spans days; each day is a fresh
    process that re-reads the JSONL checkpoint and continues. No daemon, no cron — this is
    stated plainly rather than implied, per the project's honesty standard.
  - **Checkpoint/results namespacing extended to (provider, model)**, not model alone
    (`_model_suffix`/`_ckpt_path`/`write_results` gained an optional `provider` param,
    default `"anthropic"` — EVERY existing call site and the already-pushed Haiku checkpoint
    filename are byte-for-byte unaffected). A sonnet run can never resume from ARM-C answers
    and vice versa; `_restore` now refuses on provider mismatch too.
  - **`--smoke` mode** (2 real calls, 1 per arm, no checkpoint written) shakes out a NEW
    provider/endpoint before spending real budget — served-model string, vision-content
    acceptance, and usage-field population are VERIFIED per the project's "measure, don't
    assume" rule, not assumed from Google AI Studio's/OpenRouter's documented shape.
  - **`model_tier_report.py`** (`aiqs-model-tier-report`, `make model-tier-report`): scans a
    run dir for every non-mock `vlm_crop_results*.csv`, reconstructs each variant's states as
    lightweight rows and re-runs the UNCHANGED `eval.crop_eval.evaluate_two_arm` (100% reuse,
    automatically degeneracy-guarded) to build a cross-tier table (verdict distribution /
    rubber-stamp check, escape Δ, P/S/U mix, independence, tokens/call, wall-clock from
    checkpoint timestamps). Walled off: `model_tier_report.md`, never `summary.md`.
  - **Data-training acceptance (explicit, per user directive):** free tiers may train on
    submitted inputs. Accepted because every image sent is a PUBLIC anomaly-detection
    benchmark (VisA/MVTec), never proprietary data — recorded here as the standing
    justification, not re-litigated per roster entry.
  - `configs/free_vlm_roster.example.yaml`: documentation only (never parsed by code) —
    example Google AI Studio / OpenRouter entries so a roster swap three weeks from now is a
    copy-paste, not a memory exercise. `openai>=1.0,<2` pinned (resolves cleanly against the
    existing stack; verified with `uv sync`).
  - 14 + 8 new tests (shared-helper unit tests, backend guard/usage/crop-interop/throttle,
    vlm_crop.py provider dispatch/checkpoint-provider-awareness/smoke-writes-no-checkpoint,
    model_tier_report end-to-end on synthetic mock variants). One real bug caught by the
    tests themselves: the report's glob pattern (`vlm_crop_results*.csv`) silently matched
    ZERO mock-prefixed files (`mock_vlm_crop_results...`) because glob has no "contains" mode
    without a leading wildcard — fixed to `*vlm_crop_results*.csv`. 115/115 at session end.
  - **What remains PENDING, stated plainly:** the actual PAID sonnet-4-6 headline run and any
    real ARM-C free-tier run have NOT been executed. This local Intel-mac host has no GPU, no
    installable anomalib 2.x, and no local copy of the VisA images or the anomaly maps (both
    gitignored, living only on the Kaggle session that produced them) — the detector/value
    file-interface architecture that makes local development possible ALSO means the real VLM
    call must run where the maps and images live. Every engineering risk class identified
    across the dry-run + Haiku rehearsal (crop never engaging, 529 storms, mid-run data loss,
    parse-failure lockup, silent model downgrade, spurious independence) is now closed in
    code; the ready-to-paste Kaggle cells for the sonnet-4-6 headline run were handed to the
    user in-session. This entry is the record of that boundary, not a claim that Phase 2B's
    headline evidence exists yet.

## How Phase 1 extends the eval contract

Phase 0 = detection metrics (`metrics.py`) + persistence (`results.py`). Phase 1 layers
decision metrics on the persisted per-image scores **without re-running the detector**:
`CostMatrix` (locked), `Decision` enum, `DecisionMetrics` (coverage, escalation,
overkill/escape rates on the AUTO-DECIDED subset, realized cost; importance-weightable
to a target prevalence). The headline is the **risk–coverage curve**, not AUROC. Risk
rates are conditioned on the auto-decided subset (selective-classification convention)
— this extends, not forks, the Phase-0 stub.

## Phase-1 follow-ups (logged, not done)

- **PatchCore baseline run pending.** `make baseline` (now defaults to PatchCore) will
  produce the first real-separation image scores; then `make decide` should yield the
  **positive risk-coverage headline** the 600-step EfficientAD could not deliver.
  `configs/patchcore_cpu.yaml` uses WideResNet-50-2 backbone, 10% coreset sampling,
  9 nearest neighbors — tunable. If image-AUROC is still weak on `screw`, try
  `capsule` or `bottle` (easier categories).
- **EfficientAD** configs (`default.yaml`, `baseline_cpu.yaml`) preserved. Full-budget
  EfficientAD on GPU host remains an option for inference-speed benchmarking later.
- **MVTec AD 2** dataset: not supported in anomalib 1.2.0 → using original
  **MVTec AD**. Revisit when on anomalib 2.x.
- Cross-Venn-Abers trades exact single-split Venn validity for full data usage
  (standard, slightly-conservative cross-conformal trade-off); fine here, revisit if a
  larger labelled set becomes available.
- **Phase 2:** LangGraph adjudication agent + VLM second-look + Langfuse, layered on
  this decision spine, once a usable detector exists.
