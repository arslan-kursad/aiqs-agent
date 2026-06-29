# AIQS-Agent

**Agentic Adjudication Layer for Industrial Visual Quality Inspection.**

An agentic decision + reasoning layer that sits **on top of** an off-the-shelf
anomaly detector and makes its outputs production-trustworthy. The value is in the
**decision layer** — cost-aware, calibrated, *abstaining* decisions with auditable
reasoning traces — not in the detector (a solved commodity). We optimize a business
cost function and the false-reject rate, **not** detection AUROC.

> Detailed north star, constraints, and architecture live in [CLAUDE.md](CLAUDE.md).

## Status

- **Phase 0 — detection baseline + eval backbone. ✅** Off-the-shelf
  [Anomalib](https://github.com/openvinotoolkit/anomalib) detector on one MVTec AD
  category, with a reusable evaluation backbone (image AUROC, pixel AUPRO, AUPIMO).
  Default detector is **PatchCore** (stronger image-level separation; EfficientAD
  configs retained). Write-up: [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md).
- **Phase 1 — calibrated, cost-aware, abstaining decision layer. ✅** Cross
  (out-of-fold) Venn-Abers calibration → cost-matrix `PASS/FAIL/ESCALATE` →
  risk-coverage, with prevalence (label-shift) correction, a **break-even review-cost**
  analysis, and an honesty guard. **Complete and validated (27/27 tests); no LLM/agent
  yet.** The layer reports an **operating envelope**, not a universal win:
  - **Weak detector** (600-step EfficientAD, AUROC 0.559): the guard refuses a
    false-positive headline — honest null, no separable signal.
  - **Strong detector** (PatchCore on Colab, AUROC 0.976): abstention cuts overkill
    (0.29→0.16) and drives escapes to 0, but at review cost = 1 the escalation overhead
    exceeds the savings, so a tuned threshold wins on *total* cost. Cost-aware abstention
    wins below a **break-even review cost**, and under a **realistic escape-dominant cost
    matrix** at low prevalence (shipping a defect ≫ a re-inspection). `make decide`
    reports both, with the full anti-cherry-pick sweep.
  - Machinery is independently validated on synthetic separating scores (`make sim`).
- **Phase 2A — VLM second-look on the ESCALATE bucket (first LLM). ✅ backbone.** Layers a
  single **`claude-sonnet-4-6` vision** adjudication step on top of the Phase-1 spine,
  ESCALATE-only, behind a calibrated abstain rule. Heart of the eval: raw accuracy, a
  **pre-registered error-independence** test vs the detector (`Wilson-lo[P(VLM ok | det
  wrong)] > 0.50`), bidirectional value (rescue vs escape), an effective-review-cost band,
  and K-run rule stability. Plain functions around a node-shaped `VLMState` seam (no
  LangGraph yet — `crop_fn` / `anomaly_map_path` hooks reserved for Phase 2B). Mock smoke
  is walled off; the real-data headline awaits a hard category. `make vlm`.
- **Phase 2B — MVTec AD 2 migration + the crop experiment. 🟡 in progress.**
  - **Stage 0 — pre-AD2 hygiene. ✅** Provenance cleanup (canonical runs, repaired
    `decisions.csv`); **live model-ID check** — every call served `claude-sonnet-4-6`, no
    silent downgrade; **token-budget** measured (a powered AD2 VLM run costs < ~$0.50 — not
    a constraint).
  - **Stage 1 — anomalib 2.x optional extra + AD2 datamodule + anomaly-map crop instrument.**
    Next. anomalib 2.x stays a **GPU-host-only optional extra**; the pinned local 1.2 stack
    and the pure-numpy decision layer are untouched.

## Quickstart

```bash
make install                 # uv sync (creates .venv, installs pinned stack)
make smoke                   # fast end-to-end sanity run (~10 steps)
make baseline CATEGORY=screw # full train + eval, writes results/
make decide                  # Phase-1 adjudication on the latest run
make vlm RUN=<id> [MOCK=1]   # Phase-2A VLM second-look on the ESCALATE bucket
make sim                     # SYNTHETIC machinery validation (NOT real-data evidence)
make test                    # unit tests (decision policy + VLM eval, API mocked)
```

Results: `results/metrics.csv`, per-run `summary.md`, and Phase-1
`risk_coverage{,_target}.png` + `decision_scores.csv` + `results/decisions.csv`.

## Stack & host notes

Pinned for an **Intel (x86_64) macOS** host (CPU-only): `anomalib 1.2.0`,
`torch 2.2.2`. See [CLAUDE.md](CLAUDE.md) for why.

### GPU / arm64 upgrade path (turnkey real baseline)

The Phase-1 decision layer is solid; what it needs is a detector with real
per-image separation. On a CUDA or Apple-Silicon host:

1. Drop the version caps in `pyproject.toml` and move to **anomalib 2.x + current
   torch** (the caps exist only because this Intel-mac has no newer-torch wheel).
2. Optionally switch the dataset to **MVTec AD 2** (supported in anomalib 2.x).
3. Train at the **full step budget** (`configs/default.yaml`, 70k steps) — the 600-step
   CPU budget is the root cause of the weak image-AUROC, not the category.
4. `make baseline` → `make decide`. With a separating detector the risk-coverage
   headline (overkill ↓, cost ↓ vs naive at matched escape) materialises — exactly as
   `make sim` shows on synthetic scores.

If EfficientAD's **image-level** AUROC lags, swap in **PatchCore** (also off-the-shelf
in Anomalib) — it tends to be stronger at image-level separation, which is what this
decision-layer demo needs. The decision layer is detector-agnostic: it only consumes
`image_scores.csv`.

## Layout

```
configs/        YAML configs (category is configurable)
src/aiqs/
  config.py     typed config + CLI overrides
  data.py       MVTec AD datamodule
  train.py      train EfficientAD on one category
  evaluate.py   evaluate a checkpoint, persist results
  decide.py     Phase-1 cost-aware adjudication over persisted scores
  vlm/          Phase-2A VLM second-look (abstain pipeline around a VLMState seam)
  vlm_decide.py Phase-2A entry point (aiqs-vlm)
  simulate_decision.py  synthetic machinery validation (walled off)
  eval/         evaluation backbone (metrics, persistence, decision policy, VLM eval)
scripts/        local diagnostics (e.g. verify_vlm_local.py — model-id + token pre-flight)
results/        metrics.csv + per-run summaries + Phase-1 decisions (committed)
tests/          unit tests for the decision policy / calibration / guard / VLM eval
```
