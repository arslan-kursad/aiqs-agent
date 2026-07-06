# ⚠️ REHEARSAL — claude-haiku-4-5, NOT the locked headline model (claude-sonnet-4-6)

_Model: claude-haiku-4-5_

## Two-arm full-vs-crop experiment (Phase 2B Stage 3)

- Bucket: 109 (57 good, 52 defective); K=5; diffuse (no crop, excluded from classification): 2.
- **Escape rate** A(full) 1.000 -> B(+crop) 0.962 (Δ=+0.038; fixed-by-crop 10, broken-by-crop 0 run-item pairs).
- **Escape classification** (PRE-REGISTERED rules, frozen pre-run): perception=5, semantic=235, unclassified=10 (rate 0.04; adequate=True), diffuse-excluded=10.
- **Stable-vs-flip** (A escapes): 52/52 stable-wrong (fraction 1.00) — stable-wrong escapes are invisible to K-run agreement.
- **Error-independence** (pre-registered, powered): A: YES: independent in 5/5 runs | B: YES: independent in 5/5 runs.
- **2A replication (ARM-A, descriptive):** good-rescue 1.00, defect-escape 1.00, confidence-separation AUC 0.50 (~0.5 = does not separate).
- **Tokens/call** A in/out 809/107 | B 1353/110.

