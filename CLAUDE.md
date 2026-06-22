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
- Results: `results/metrics.csv` (one row per eval) + `results/runs/<id>/`
  (`summary.md`, `image_scores.csv`) are **committed** so configs compare over
  time. Datasets, checkpoints, and heavy per-pixel dumps are gitignored.
- The eval backbone (`src/aiqs/eval/`) is the spine every later phase is measured
  against; `eval/decision.py` is the Phase-1 decision-metrics contract (stub now).

## Current phase

**Phase 0 — foundation + detection baseline + eval skeleton. ✅ COMPLETE.**
No LLM, no agent.

- [x] uv project (src layout, pyproject, committed lockfile), Makefile, README,
      .gitignore, `git init`.
- [x] Pinned, verified-installable Intel-mac detector stack.
- [x] Train EfficientAD on one MVTec AD category (default: `screw`).
- [x] Eval backbone: image AUROC, pixel AUPRO, AUPIMO (all working in 1.2.0).
- [x] Persist results + print baseline summary.

**Baseline (reduced budget, 600 steps, CPU, `configs/baseline_cpu.yaml`):**
pixel AUROC **0.940** · pixel AUPRO **0.821** · pixel AUPIMO 0.0023 · image AUROC
**0.559** · image F1 0.685. Strong localisation, weak image-level separation — the
untrustworthy-detector case the Phase-1 layer targets. Full 70k-step run deferred to
a GPU/arm64 host (`configs/default.yaml`). See `docs/PHASE0_REPORT.md`.

Default category: **`screw`** (small/subtle defects → many borderline scores →
good material for the Phase-1 adjudication layer). Fully configurable.

**Next: Phase 1** — calibration + LangGraph adjudication agent + cost-matrix
PASS/FAIL/ESCALATE, measured on the persisted `image_scores.csv`.

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

## Phase-1 follow-ups (logged, not done)

- **MVTec AD 2** dataset: not supported in anomalib 1.2.0 → using original
  **MVTec AD**. Revisit when on anomalib 2.x.
- Re-run the baseline at full step budget on an arm64/CUDA host (anomalib 2.x).
- Wire `eval/decision.py`: calibration + cost-matrix PASS/FAIL/ESCALATE, computing
  false-reject rate / escalation rate / decision cost from persisted scores.
