# Model-tier comparison (Phase 2B) — NOT the canonical headline document

One row per VLM tier tested on this bucket (Haiku rehearsal, the claude-sonnet-4-6 headline, any ARM-C free-tier run, ...). See summary.md for the canonical (sonnet-4-6) result; this file is a cross-tier comparison only.

| model                          | provider         | escape A | escape B |       Δ | P/S/U        | indep A  | indep B  |   tok/call A |   tok/call B | wall(min) |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| claude-sonnet-4-6              | anthropic        |    0.500 |    0.115 |  +0.385 | 48/24/58     | YES      | YES      |      810/101 |     1354/273 |     111.5 |

## Per-model detail

### claude-sonnet-4-6 (provider: anthropic, source: vlm_crop_results.csv)

- **Verdict distribution** (rubber-stamp check): A: clean=77%, unsure=12%, defect=11% | B: clean=51%, defect=33%, unsure=17%
## Two-arm full-vs-crop experiment (Phase 2B Stage 3)

- Bucket: 109 (57 good, 52 defective); K=5; diffuse (no crop, excluded from classification): 0.
- **Escape rate** A(full) 0.500 -> B(+crop) 0.115 (Δ=+0.385; fixed-by-crop 102, broken-by-crop 2 run-item pairs).
- **Escape classification** (PRE-REGISTERED rules, frozen pre-run): perception=48, semantic=24, unclassified=58 (rate 0.45; adequate=False), diffuse-excluded=0.
- **Stable-vs-flip** (A escapes): 26/26 stable-wrong (fraction 1.00) — stable-wrong escapes are invisible to K-run agreement.
- **Error-independence** (pre-registered, powered): A: YES: independent in 5/5 runs | B: YES: independent in 5/5 runs.
- **2A replication (ARM-A, descriptive):** good-rescue 0.95, defect-escape 0.50, confidence-separation AUC 0.49 (~0.5 = does not separate).
- **Tokens/call** A in/out 810/101 | B 1354/273.

