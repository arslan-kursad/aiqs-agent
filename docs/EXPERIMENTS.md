# Experiment log & evidence

Every claim below is backed by a committed artifact under `results/` (or an explicit
honest-null entry in [CLAUDE.md](../CLAUDE.md)). Mock/synthetic outputs are walled off
and never cited.

## 1. Phase 0 — detection baselines

| Detector | Ground | image-AUROC | Note |
|---|---|---|---|
| EfficientAD (600-step CPU) | MVTec `screw` | 0.559 | deliberately weak — the untrustworthy-detector case |
| PatchCore (GPU) | MVTec `screw` | 0.976 | saturated |
| PatchCore (GPU) | MVTec `capsule` | 0.976 | saturated |
| PatchCore (GPU) | VisA six-category sweep | 0.646–0.972 | see §3 |

## 2. Phase 1 — the operating envelope

The decision layer (cross Venn-Abers → locked cost matrix → PASS/FAIL/ESCALATE) is
compared against the **cost-optimal tuned threshold** on the same scores. Three measured
regimes:

| Regime | Ground | Outcome |
|---|---|---|
| Weak detector | EfficientAD screw (0.559) | **honest null** — guard refuses a false headline; no policy can help |
| Strong detector, expensive review | PatchCore screw (0.976) | abstention cuts overkill 0.29→0.16 and escapes→0, but review overhead loses on total cost — threshold suffices; break-even review cost reported |
| Genuine uncertainty | **VisA candle (0.972)** | **first real win: 11% (native) / 13% (realistic 100/3/1) cheaper than the tuned threshold** |

Per-run evidence: `results/runs/<id>/risk_coverage.png`, `risk_coverage_breakeven.png`,
`breakeven.csv`, `summary.md`. Example (VisA candle → capsules runs are committed):

![risk-coverage](../results/runs/patchcore-wide_resnet50_2_visa-capsules_20260705T181237Z/risk_coverage.png)

## 3. Phase 2B Stage 2 — the substrate hunt (VisA sweep)

Standard MVTec saturates → the ESCALATE bucket empties → nothing to adjudicate. The gate
(pre-registered): ESCALATE∩good **and** n_dw ≥ ~30. VisA sweep (Kaggle GPU, PatchCore,
identical config/seed — candle numbers reproduced exactly across two hosts):

| category | image-AUROC | bucket | ESC∩good | n_dw | verdict |
|---|---|---|---|---|---|
| candle | 0.972 | 39 | 28 | 11 | direction-only |
| **capsules** | **0.739** | **109** | **57** | **54** | **powered — Stage-3 ground** |
| **macaroni1** | **0.815** | **138** | **83** | **45** | powered (conditional second ground) |
| macaroni2 | 0.646 | 174 | 92 | 80 | rejected — 87% escalation ≈ VLM-on-everything |
| pcb1 | 0.936 | 65 | 35 | 26 | borderline |
| pcb2 | 0.928 | 74 | 49 | 29 | borderline |

## 4. Phase 2B Stage 3 — two-arm full-vs-crop

**Design (locked before data):** same 109-item ESCALATE bucket, ARM-A full-image vs
ARM-B full+crop, K=5, arm-independent single-turn calls, pre-registered escape
classification (`src/aiqs/vlm/reasoning_rules.py`, frozen 2026-07-02), served-model stop,
checkpoint/resume.

**Dry-run #1 (voided, $5 lesson).** The crop never engaged: anomalib-2.x maps carry a
high normalization floor and the flat-map guard mislabeled 19/20 real maps as diffuse.
ARM-B ran byte-identical to ARM-A. Fixed against the real maps (quantile + geometric
bbox diffuse test → 19/20 crop), institutionalized as: *dry-run the instrument on real
exported maps before spending API budget.*

**Haiku rehearsal (complete — $1.77, 1090/1090 calls).** Run with
`--model claude-haiku-4-5` in a contamination-proof rehearsal namespace (the locked
headline model is claude-sonnet-4-6):

| Measure | ARM A (full) | ARM B (+crop) |
|---|---|---|
| verdicts | "clean" **545/545** | 535 clean · 5 defect · 5 unsure |
| escape rate (defectives) | 1.000 | 0.962 |
| tokens/call (in) | 809 | 1353 (+544 = the crop block) |

- Escape classification (250 eligible): **perception 5 (2%) · SEMANTIC 235 (94%) ·
  unclassified 10 (4% — labeling adequate)**.
- Escapes 100% stable-wrong (52/52) → K-run agreement is not an abstain signal.
- Confidence separation AUC 0.50 → self-reported confidence carries no signal.
- 5 parse failures handled by the loud fallback (marked, non-blocking) — the resilience
  layer worked in production.

**Rehearsal-grade conclusion:** a cheap-tier VLM second-look is a *rubber stamp* on
borderline industrial images, and its failures are **semantic, not perceptual** — it sees
the flagged region and calls it normal variation. Better pixels (the crop) do not fix a
semantic failure.

## 5. Degeneracy guard (added before the headline run)

The Haiku rehearsal's "independent in 5/5 runs" was formally correct and substantively
meaningless — a 545/545 rubber stamp satisfies `Wilson-lo > 0.50` by luck of being "right"
on whichever side the detector over-rejects. `eval/vlm_eval.py` now forces
`"invalid-degenerate"` whenever one raw verdict covers ≥95% of a run
(`DEGENERATE_VERDICT_FRAC`, pre-registered before any headline data exists). This applies
to **every** tier, including the not-yet-run sonnet-4-6 headline — full rationale and the
tie-break bugfix it surfaced are in [CLAUDE.md](../CLAUDE.md) (2026-07-06 entries).

## 6. ARM-C — the model-tier lever

A provider-agnostic `OpenAICompatibleVLMBackend` (Google AI Studio, OpenRouter, ...) reuses
the identical bucket, crop instrument, checkpoint/resume, and served-model guard as the
Anthropic path — swapping a free-tier roster entry is a config change (`--base-url
--model --api-key-env`), never a code change. See
[`configs/free_vlm_roster.example.yaml`](../configs/free_vlm_roster.example.yaml) and
CLAUDE.md for the full design (rate-limit/resume/data-training-acceptance rationale). No
real ARM-C run has been executed yet — engineering only, pending a GPU/data-bearing host.

## 7. Model-tier comparison (fill in once real runs exist)

`aiqs-model-tier-report` (`make model-tier-report RUN=<id>`) scans a run dir for every
real `vlm_crop_results*.csv` and reconstructs the full two-arm evaluation per tier —
verdict distribution (rubber-stamp check), escape Δ, P/S/U classification, the
degeneracy-guarded independence rule, tokens/call, and wall-clock. Table below is a
**template** — fill with real numbers as tiers complete, never with placeholders passed
off as data:

| model | provider | escape A→B Δ | P/S/U | independence A / B | tokens/call A / B | wall-clock |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | anthropic | 1.000→0.962 (Δ+0.038) | 5/235/10 | invalid-degenerate (rubber stamp) both arms | 809/107 → 1353/110 | ~minutes (single Kaggle session) |
| claude-sonnet-4-6 | anthropic | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| (ARM-C free-tier) | openai_compatible | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

## 8. Next — the locked headline run

`claude-sonnet-4-6`, same ground, same frozen rules, ~$5, fully resume-safe (checkpoint
survives a crash or a killed Kaggle session — re-running the same command continues
without re-billing a completed call). The rehearsal sharpens the question it must answer:
**are sonnet's escapes also semantic-dominated?** If yes, the Phase-2B lever is
prompt/anchor design, not image fidelity — and that redirects the roadmap.

## 9. Stage-4 verdict — template (fill with numbers, do not re-narrate after seeing them)

Exactly one of the three frames below gets filled in once the sonnet-4-6 headline (and,
if triggered, the macaroni1 second ground) completes. The frame is chosen by the data, not
the other way around — this section exists precisely so the verdict isn't re-litigated
after the numbers are known.

**Frame A — Positive.** *"Sonnet's escapes are NOT rubber-stamp/semantic-dominated: escape
rate moved [X]→[Y] (Δ=[D]) with the crop, PERCEPTION classified at [P]% (vs Haiku's 2%),
Wilson-lo=[W] (powered, n_dw=54) exceeds 0.50 non-degenerate. The full-vs-crop hypothesis
is CONFIRMED at the headline tier: a second look with the right image adds independent
signal beyond the detector."* → writeup + final commit + merge.

**Frame B — Null / semantic-dominant (the Haiku pattern replicates).** *"Sonnet's escapes
are ALSO semantic-dominated: [S]% classify SEMANTIC (vs Haiku's 94%), crop moves escape by
only [D] (vs Haiku's 0.038), independence is [invalid-degenerate / theatre] at the
headline tier too. The Phase-2B lever is NOT image fidelity — better pixels do not fix a
model that sees the flagged region and calls it normal. The lever is prompt/anchor design
or a fundamentally different verification strategy."* → informative negative result,
writeup + merge (pre-registered criteria make this a real finding, not a failure).

**Frame C — Partial.** *"Crop reduced escape by [D], below the pre-registered
[threshold]; independence is [redundant/independent] but [caveat]. Cost-aware abstention
gains a bounded, non-zero lever — a new operating-envelope boundary point, not a clean
win."* → envelope update + next-step proposal (prompt/few-shot iteration), not a final
verdict either way.
