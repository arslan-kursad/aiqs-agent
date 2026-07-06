"""Phase-2A: VLM second-look on the ESCALATE bucket — the first LLM component.

This package layers a vision-LLM adjudication step ON TOP of the deterministic
Phase-1 decision spine (``aiqs.eval.decision``). It runs ONLY on items the Phase-1
policy escalated; the Tier-1 PASS/FAIL path is never touched.

Design (see CLAUDE.md decision log, 2026-06-23):
  * Plain functions around a node-shaped seam (``VLMState`` + ``adjudicate``). NO
    LangGraph in 2A — a single-node/single-pass graph does not justify the dependency
    (violates "no premature abstraction"); 2B wraps ``adjudicate`` as a node mechanically.
  * Single model: claude-sonnet-4-6 vision. NO Opus tier (a separately-measured 2B lever).
  * Calibrated abstain RULE (cost-matrix expected-cost argmin, reusing Phase-1
    ``decide_one``), but the vlm_confidence -> p_vlm map is PROVISIONAL/UNCALIBRATED in
    2A (no in-bucket data to fit Venn-Abers); full VLM calibration is 2B.
  * Single-pass (VLM -> human on abstain); no recursive re-perception loop (that is 2B).
"""

from aiqs.vlm.abstain import adjudicate_probability, confidence_to_p
from aiqs.vlm.adjudicate import adjudicate
from aiqs.vlm.backend import (
    AnthropicVLMBackend,
    MockVLMBackend,
    VLMVerdict,
    parse_verdict,
)
from aiqs.vlm.crop_fn import make_crop_fn
from aiqs.vlm.state import VLMState
from aiqs.vlm.substrate import SubstrateError, bucket_composition, substrate_guard

__all__ = [
    "VLMState",
    "VLMVerdict",
    "parse_verdict",
    "AnthropicVLMBackend",
    "MockVLMBackend",
    "confidence_to_p",
    "adjudicate_probability",
    "adjudicate",
    "make_crop_fn",
    "bucket_composition",
    "substrate_guard",
    "SubstrateError",
]
