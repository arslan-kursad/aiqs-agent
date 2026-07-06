"""Calibrated abstain rule for the VLM second-look.

Two pieces, kept deliberately small:

1. ``confidence_to_p`` — map a (verdict, confidence) pair to a PROVISIONAL P(defective).
   This is explicitly UNCALIBRATED in 2A: LLMs are overconfident, and the ESCALATE
   bucket is too small to fit a Venn-Abers calibrator on VLM votes (that is a 2B
   deliverable). A shrinkage knob ``lam`` pulls the probability toward 0.5; the eval
   sweeps it to report the break-even shift as a SENSITIVITY BAND rather than a single
   number, and LEADS from the conservative (high-shrinkage) end.

2. ``adjudicate_probability`` — the abstain decision. The VLM token cost is already sunk
   once the model is called, so the only trade-off left is VLM-error-risk vs the human
   review cost. That is EXACTLY the Phase-1 expected-cost argmin applied to ``p_vlm``,
   where the ESCALATE action's cost IS the human review cost: auto-decide (PASS/FAIL)
   iff it is cheaper than review, else ESCALATE -> human. So we reuse ``decide_one``
   verbatim — no second policy to keep in sync, and the same safety-first tie-break
   (never silently PASS an uncertain item) carries over.
"""

from __future__ import annotations

from aiqs.eval.decision import CostMatrix, Decision, decide_one


def confidence_to_p(verdict: str, confidence: float, lam: float = 0.0) -> float:
    """Provisional, UNCALIBRATED P(defective) from a VLM verdict + confidence.

    defect -> 0.5 + 0.5c, clean -> 0.5 - 0.5c, unsure -> 0.5 (always abstains under the
    locked matrix). ``lam`` in [0, 1] shrinks toward 0.5 (lam=1 => always 0.5 => always
    abstain). Confidence is clamped to [0, 1] defensively.
    """
    c = min(1.0, max(0.0, float(confidence)))
    if verdict == "defect":
        p = 0.5 + 0.5 * c
    elif verdict == "clean":
        p = 0.5 - 0.5 * c
    elif verdict == "unsure":
        p = 0.5
    else:
        raise ValueError(f"unknown verdict {verdict!r} (expected defect/clean/unsure)")
    lam = min(1.0, max(0.0, float(lam)))
    return 0.5 + (1.0 - lam) * (p - 0.5)


def adjudicate_probability(p_vlm: float, cost: CostMatrix) -> Decision:
    """Abstain rule = Phase-1 policy on p_vlm. ESCALATE means 'abstain -> human'.

    auto-decide (PASS/FAIL) iff min(E[PASS], E[FAIL]) < review cost; else ESCALATE.
    Under the realistic escape-dominant matrix (100/3/1) this self-floors auto-PASS of
    a likely defect: PASS is chosen only if p_vlm < review/escape (= c_human/100), so a
    wrong-confident clean verdict cannot cheaply ship a defect.
    """
    return decide_one(p_vlm, cost)
