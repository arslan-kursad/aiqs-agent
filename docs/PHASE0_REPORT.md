# AIQS-Agent — Phase 0 Report

**Agentic Adjudication Layer for Industrial Visual Quality Inspection**

| | |
| --- | --- |
| **Phase** | 0 — foundation + detection baseline + evaluation skeleton |
| **Date** | 2026-06-22 |
| **Repository** | https://github.com/arslan-kursad/aiqs-agent (private) |
| **Status** | Pipeline complete & validated end-to-end; real baseline number pending a training-budget decision |
| **Host** | Intel Core i5-5350U (2c @ 1.8 GHz), macOS 12.7.6, x86_64 — **CPU-only** |

---

## 1. Executive summary

Phase 0 stands up the project and a rigorous, reusable evaluation harness around an
**off-the-shelf** anomaly detector (Anomalib EfficientAD) on a single MVTec AD
category. No LLM and no agent yet — that is deliberate. The thesis of the project is
that the production value lives in the **decision/adjudication layer** (cost-aware,
calibrated, abstaining decisions with auditable traces), *not* in the detector, which
is a solved commodity. Phase 0 therefore invests almost entirely in (a) a clean,
reproducible foundation and (b) an evaluation backbone designed to be extended in
Phase 1 with decision-level metrics.

The full pipeline — **data preparation → training → evaluation → persisted results** —
has been validated end-to-end on the host CPU. Five distinct real-world
incompatibilities were discovered and resolved along the way (see §6). The only
remaining Phase-0 item is producing a *meaningful* baseline number, which requires a
real (non-trivial) training run; on this CPU-only host that is a time/throughput
decision rather than an engineering one (see §9).

---

## 2. North star & thesis (context)

> The value in industrial visual inspection is **not** in the detector (commodity,
> solved) but in the **decision layer** — reducing false rejects (overkill) via
> cost-aware, calibrated, **abstaining** decisions with auditable reasoning traces.
> We optimize a **business cost function** and the **false-reject rate**, not detection
> AUROC.

Target architecture (built incrementally, not all at once):

```
Detector (Anomalib)
  → calibration
  → adjudication agent (LangGraph): rules + VLM second-look (Claude API)
                                    + later similar-case retrieval
  → cost-matrix decision policy: PASS / FAIL / ESCALATE(→human)
  → later: root-cause agent, memory, conversational copilot, FastAPI
```

Phase 0 delivers only the leftmost box plus the measurement spine everything later is
judged against.

---

## 3. Scope & deliverables

| # | Phase-0 requirement | Status |
| - | ------------------- | ------ |
| 1 | Clean uv project (src layout, pyproject, lockfile, README, .gitignore, Make targets `install`/`train`/`eval`) | ✅ Done |
| 2 | `CLAUDE.md` capturing north star, constraints, architecture, stack, current phase | ✅ Done |
| 3 | Anomalib EfficientAD on one MVTec AD category, category configurable | ✅ Done (default `screw`) |
| 4 | Reusable eval module: image AUROC, pixel AUPRO, AUPIMO (if cleanly supported); persist CSV + markdown; designed for Phase-1 decision metrics | ✅ Done |
| 5 | Print a clear baseline summary | ✅ Done |
| — | **Meaningful baseline number** (real training run) | ⏳ Pending decision (§9) |

---

## 4. What was built

```
aiqs-agent/
├── pyproject.toml            # pinned stack, console scripts
├── uv.lock                   # committed for reproducibility
├── Makefile                  # install · data · train · eval · baseline · smoke
├── README.md
├── CLAUDE.md                 # durable project memory (read first each session)
├── configs/default.yaml      # category + model + training config (all CLI-overridable)
├── docs/PHASE0_REPORT.md      # this report
└── src/aiqs/
    ├── config.py             # typed config (dataclasses + YAML) + CLI overrides
    ├── prepare_data.py       # MVTec category fetch/reorg + synthetic ImageNette
    ├── data.py               # anomalib MVTec datamodule
    ├── detector.py           # EfficientAD model + train/eval Engine factories
    ├── train.py              # entry point: aiqs-train
    ├── evaluate.py           # entry point: aiqs-eval (test + score collection)
    └── eval/                 # ← the measurement spine
        ├── metrics.py        # detection metrics config + canonicalisation
        ├── results.py        # persistence: metrics.csv + summary.md + image_scores.csv
        └── decision.py       # Phase-1 decision-metrics contract (stub)
```

**Design choices of note**

- **Minimal config stack** — plain nested dataclasses + PyYAML + argparse. No
  omegaconf/pydantic; every field is CLI-overridable (`--category`, `--max-steps`,
  `--imagenet-dir`, …). The category is fully configurable.
- **Persist raw per-image scores, not just aggregates.** `evaluate.py` attaches a
  Lightning callback that captures `(image_path, gt_label, anomaly_score)` for every
  test image and writes `image_scores.csv`. This is the deliberate bridge to Phase 1:
  the adjudication layer (calibration + cost-matrix PASS/FAIL/ESCALATE) can be built
  and measured on these scores **without re-running the detector**.
- **Detector is off-the-shelf.** No custom architecture is defined anywhere — we only
  configure anomalib's `EfficientAd` and its `Engine`, per the hard constraints.

---

## 5. The stack & the Intel-mac constraint

The host is a 2015 **Intel (x86_64)** Mac. Two consequences shaped every version pin:

1. **No GPU acceleration** — Metal/MPS is Apple-Silicon-only and there is no CUDA, so
   training and inference run on a 2-core CPU.
2. **PyTorch wheel ceiling** — official PyTorch macOS wheels are **arm64-only from
   torch 2.3 onward**; the last x86_64-macOS torch is **2.2.2**. The latest anomalib
   (2.5) requires a newer torch with no installable x86_64-macOS wheel, so it **cannot
   be installed on this host**.

The resolution: pin the **anomalib 1.2 line** against torch 2.2.2. Critically, AUPIMO
was added in anomalib 1.1, so this pin keeps **every** Phase-0 metric — nothing is
sacrificed on the metrics side.

Verified-installable, contemporaneous pin (Python 3.11.15, via `uv`):

| package | version | reason |
| ------- | ------- | ------ |
| anomalib | 1.2.0 (`[core,vlm]`) | last line installable here; AUPIMO present |
| torch | 2.2.2 | last x86_64-macOS wheel |
| torchvision | 0.17.2 | pairs with torch 2.2.2 |
| numpy | 1.26.4 | `<2` — torch 2.2.2 uses the NumPy 1.x ABI |
| lightning | 2.4.0 | capped `<2.5` (anomalib 1.2 leaves it unbounded) |
| torchmetrics | 1.4.3 | capped `<1.5` (same reason) |
| pandas | 2.2.3 | capped `<2.3` (avoid pandas 3.0 breakage) |
| matplotlib | 3.9.4 | capped `<3.10` (anomalib viz uses removed `tostring_rgb`) |
| ollama | 0.3.3 | capped `<0.4` (anomalib VLM backend uses removed `_encode_image`) |

**Upgrade path:** on an arm64 (Apple Silicon) or CUDA host, drop these caps and move to
**anomalib 2.x + current torch**. The Phase-1+ stack (LangGraph, VLM, FastAPI,
Langfuse) is independent of this detector pin.

---

## 6. Obstacles encountered & resolutions

Every one of these was discovered empirically and resolved transparently; none were
silently worked around.

| # | Symptom | Root cause | Resolution |
| - | ------- | ---------- | ---------- |
| 1 | Latest anomalib/torch won't install | No x86_64-macOS torch wheel past 2.2.2 | Pinned anomalib 1.2.0 + torch 2.2.2 (keeps AUPIMO) |
| 2 | `import anomalib.{data,models}` fails on `sklearn`, `dotenv`, `requests`; `ollama._encode_image` ImportError | anomalib 1.2.0 under-declares deps **and** eagerly imports every model (incl. VlmAd) | Added `scikit-learn`, `requests`, `[vlm]` extra; pinned `ollama<0.4` |
| 3 | MVTec download → **HTTP 404** | anomalib hard-codes a dead mydrive.ch URL | `prepare_data.py` fetches the requested category from the public HF mirror `TheoM55/mvtec_anomaly_detection` and reorganises it into anomalib's native layout |
| 4 | EfficientAD wants to pull **~1.5 GB** ImageNette | Penalty/distillation dataset download | Smoke test uses a tiny **synthetic** ImageNette (`--imagenet-dir`); real ImageNette reserved for real baselines |
| 5 | Eval: `RuntimeError: Missing key(s) … pixel_metrics.AUPRO.fpr_limit` | Eval-time metric buffers absent from the training checkpoint; `Engine.test(ckpt_path=…)` loads strictly | Load weights manually with `strict=False` (metric buffers are accumulators, not learned weights) |
| 6 | Eval: `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` | matplotlib 3.10 removed `tostring_rgb`; anomalib's viz callback still calls it | Pinned matplotlib `<3.10` |

---

## 7. Evaluation backbone (the spine)

`src/aiqs/eval/` is intentionally decoupled from torch/anomalib (plain pandas + stdlib)
so it can be reused and extended across phases.

**Phase-0 outputs (per eval run):**

```
results/
├── metrics.csv                       # one appended row per run (metadata + metrics)
└── runs/<run_id>_<utc-timestamp>/
    ├── summary.md                    # human-readable metrics + notes
    ├── image_scores.csv              # per-image (path, label, score) → Phase-1 input
    └── config.yaml                   # exact config used (provenance)
```

**Detection metrics:** image-level AUROC + F1Score, pixel-level AUROC + AUPRO, and
**AUPIMO** on a best-effort basis. AUPIMO is attempted first; if it cannot compute
cleanly it is dropped, the run proceeds with AUROC/AUPRO, and a note is recorded in
`summary.md` and the console — matching the project rule of *fall back explicitly, do
not silently degrade.*

**Phase-1 extension point:** `eval/decision.py` defines (as a stub) the decision-level
contract — a `CostMatrix`, a `Decision` enum (PASS/FAIL/ESCALATE), and a
`DecisionMetrics` shape (false-reject rate, false-accept rate, escalation rate, total
and per-item cost). Phase 1 implements `evaluate_decisions(...)` over the persisted
`image_scores.csv`, appending decision metrics to the same run directory.

---

## 8. End-to-end validation (smoke test)

`make smoke` runs the entire pipeline with a 10-step training budget and the synthetic
ImageNette. It executed cleanly end to end on the host CPU:

- **Data prep** — idempotent; reused the prepared `screw` category.
- **Training** — EfficientAD trained, checkpoint written.
- **Evaluation** — ran `test()`, collected per-image scores, persisted all artifacts,
  printed the baseline summary.

The metric *values* are degenerate **by design** — a 10-step model outputs an almost
constant anomaly map, so every score collapses to 0.5. This is exactly what validates
the harness rather than the model:

| metric | smoke value | interpretation |
| ------ | ----------- | -------------- |
| image_auroc | 0.5000 | random — untrained model |
| image_f1score | 0.0000 | random — untrained model |
| pixel_auroc | 0.5000 | random — untrained model |
| pixel_aupro | nan | undefined on a constant map |
| pixel_aupimo | *fallback* | `ValueError: Invalid threshold bounds … 0.5 <= 0.5` → dropped & logged (fallback works) |

160 per-image scores were collected into `image_scores.csv`. The AUPIMO fallback firing
on a degenerate model is the **intended** behaviour and confirms the graceful-fallback
path. On a properly trained model, the anomaly maps are non-constant and these metrics
become meaningful.

> Note: the eval pass took ~13 min on CPU, dominated by anomalib's per-image
> visualization rendering (which Phase 0 does not need). Disabling that callback is a
> planned quick win before the real baseline (est. eval ≈ 2 min).

---

## 9. Current state & the open decision

**Everything is in place to produce a real baseline.** What remains is a
resource/throughput decision, not an engineering one, because the host is CPU-only:

- A real baseline needs the **~1.5 GB** real ImageNette.
- Paper-grade EfficientAD training is **70 000 steps**; on a 2-core 1.8 GHz CPU that is
  impractical (many hours to days).

Three options are on the table:

- **A — Time-boxed local baseline:** measure CPU step-rate, choose a budget for a ~1–2 h
  run, download ImageNette, train + eval, commit real numbers. Yields a genuine (if not
  paper-grade) baseline this session.
- **B — Defer to a GPU/arm64 host:** the pipeline is ready; run
  `make baseline CATEGORY=screw` there at the full step budget for the definitive number.
- **C — Quick non-degenerate run:** a short local run (~1 k steps) just to populate
  `results/` with real-data numbers; weak but non-trivial.

This report will be updated with the baseline table once the run completes.

---

## 10. Phase-1 follow-ups (logged)

- **MVTec AD 2** dataset and a refreshed AUPIMO — not supported on the anomalib 1.2
  line; revisit on anomalib 2.x.
- Re-run the baseline at the full step budget on an arm64/CUDA host.
- Disable the eval visualization callback to cut eval wall-time on CPU.
- Wire `eval/decision.py`: calibration + cost-matrix PASS/FAIL/ESCALATE, computing
  false-reject / escalation / decision-cost from the persisted per-image scores.

---

## 11. How to run / reproduce

```bash
make install                     # uv sync — creates .venv from the committed lockfile
make smoke                       # fast end-to-end wiring check (synthetic ImageNette)
make baseline CATEGORY=screw     # full: prepare data → train → eval → results/
# category is configurable, e.g. CATEGORY=transistor / capsule / hazelnut
```

Outputs land in `results/` (committed) and `models/` (gitignored). Datasets are
fetched into `datasets/` (gitignored) automatically.

---

## 12. Appendix

- **Dataset (screw):** 320 train/good · 160 test (good + 5 defect types) · 119 GT masks.
- **Dataset source:** HF mirror `TheoM55/mvtec_anomaly_detection`, category-only fetch.
- **Initial commit:** `eaed9b9` — "Phase 0: foundation + EfficientAD detection baseline
  + eval backbone".
- **Entry points:** `aiqs-prepare-data`, `aiqs-train`, `aiqs-eval`.
