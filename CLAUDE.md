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
