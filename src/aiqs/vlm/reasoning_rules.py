"""PRE-REGISTERED escape-reasoning classification — COMMITTED BEFORE the Stage-3 run.

These rules label WHY an ARM-A escape happened (a defective the full-image VLM auto-PASSed),
using the SAME item's ARM-B (full+crop) outcome. They are rule-based (no LLM judge) and
FROZEN before any real Stage-3 data exists, for the same reason the Wilson-lo threshold was
frozen in 2A: writing the keyword list AFTER reading the escapes' reasonings would let the
rule drift toward the desired result (p-hacking via labeling). Committed 2026-07-02, before
the first real capsules run. DO NOT extend these rules after seeing data; if they prove
inadequate, that is a FINDING (report it), not a license to revise.

Classification unit: one (run, item) pair where the item is a bucket DEFECTIVE (label=1)
and ARM-A auto-PASSed it (escape). Items whose anomaly map was DIFFUSE (no crop, ARM-B is
byte-identical to ARM-A) are EXCLUDED from the denominator and counted separately — a
same-input comparison cannot separate perception from semantics.

Priority order (first match wins):
  1. PERCEPTION  — ARM-B verdict on the same item is "defect": the crop made the defect
     visible, so ARM-A's failure was perceptual (it could not see it at full-image scale).
     Objective (verdict flip), no text matching involved.
  2. SEMANTIC    — ARM-B verdict is still "clean" AND its reasoning matches a NORMALIZING
     pattern below: the model engages with the flagged appearance and judges it acceptable
     — a semantic failure that better pixels do not fix.
  3. UNCLASSIFIED — everything else ("unsure" in ARM-B, empty reasoning, no pattern match,
     parse anomalies). Reported as its own rate. If UNCLASSIFIED > 0.30 of classified-
     eligible escapes, the pre-registered verdict is "rule-based labeling insufficient —
     human read required"; the rules are NOT to be widened post hoc.

NORMALIZING patterns (case-insensitive regex, word-boundary guarded — note ``\\bnormal\\b``
does NOT match inside "abnormal"):
  * artifact attributions: reflection / glare / lighting / illumination / shadow / dust /
    debris / smudge / artifact / artefact
  * explicit normalization: "normal variation|appearance|surface|texture|feature",
    "within (normal|acceptable|tolerance)", "(acceptable|expected|typical)
    (variation|appearance|feature|surface)", cosmetic, harmless
"""

from __future__ import annotations

import re

PERCEPTION = "perception"
SEMANTIC = "semantic"
UNCLASSIFIED = "unclassified"
DIFFUSE_EXCLUDED = "diffuse_excluded"

#: Pre-registered UNCLASSIFIED ceiling; above it the labeling itself is declared inadequate.
UNCLASSIFIED_MAX = 0.30

_NORMALIZING_PATTERNS = [
    r"\b(reflection|glare|lighting|illumination|shadow|dust|debris|smudge|artifact|artefact)\b",
    r"\bnormal (variation|appearance|surface|texture|feature)\b",
    r"\bwithin (normal|acceptable|tolerance)\b",
    r"\b(acceptable|expected|typical) (variation|appearance|feature|surface)\b",
    r"\bcosmetic\b",
    r"\bharmless\b",
]
_NORMALIZING_RE = re.compile("|".join(_NORMALIZING_PATTERNS), re.IGNORECASE)


def classify_escape(arm_b_verdict: str | None, arm_b_reasoning: str | None,
                    *, diffuse: bool) -> str:
    """Classify ONE ARM-A escape by its ARM-B outcome. Pure + deterministic (auditable)."""
    if diffuse:
        return DIFFUSE_EXCLUDED
    if arm_b_verdict == "defect":
        return PERCEPTION
    if arm_b_verdict == "clean" and arm_b_reasoning and _NORMALIZING_RE.search(arm_b_reasoning):
        return SEMANTIC
    return UNCLASSIFIED


def distribution(labels: list[str]) -> dict:
    """Counts + the UNCLASSIFIED rate over classification-ELIGIBLE items (diffuse excluded),
    plus the pre-registered adequacy verdict."""
    counts = {k: labels.count(k) for k in (PERCEPTION, SEMANTIC, UNCLASSIFIED, DIFFUSE_EXCLUDED)}
    eligible = counts[PERCEPTION] + counts[SEMANTIC] + counts[UNCLASSIFIED]
    rate = counts[UNCLASSIFIED] / eligible if eligible else float("nan")
    counts["unclassified_rate"] = rate
    counts["labeling_adequate"] = (rate <= UNCLASSIFIED_MAX) if eligible else None
    return counts
