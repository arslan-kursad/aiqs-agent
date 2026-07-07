"""Pydantic request/response schemas for the FastAPI Intelligent API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CostMatrixOverride(BaseModel):
    model_config = {"extra": "forbid"}

    false_accept: float | None = Field(default=None, description="escape cost override")
    false_reject: float | None = Field(default=None, description="overkill cost override")
    escalation: float | None = Field(default=None, description="review cost override")


class AdjudicateRequest(BaseModel):
    model_config = {"extra": "forbid"}

    item_id: str | None = Field(default=None, description="client-supplied; else a uuid4 is generated")
    anomaly_score: float = Field(description="raw detector anomaly score")
    image_path: str | None = Field(
        default=None, description="path relative to the server's configured --image-root; "
                                  "rejected if --image-root is unset or the path escapes it")
    image_b64: str | None = Field(default=None, description="base64-encoded image bytes (safe default)")
    target_prevalence: float | Literal["native"] | None = Field(
        default=None,
        description="production defect-rate to prior-shift to; 'native' = no shift "
                    "(use the run's own sample prevalence); omitted = the run's default (0.02)")
    cost_matrix: CostMatrixOverride | None = Field(
        default=None, description="override any of the run's LOCKED cost-matrix fields")
    lam: float = Field(default=0.0, ge=0.0, le=1.0,
                       description="VLM provisional-p shrinkage toward 0.5 (0=trust confidence)")


class HumanVerdictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    decision: Literal["pass", "fail"]
    reviewer: str | None = None
    note: str | None = None


class VLMInfo(BaseModel):
    fired: bool
    verdict: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class AdjudicateResponse(BaseModel):
    item_id: str
    decision: str | None = Field(description="'pass' | 'fail' | 'pending_human'")
    resolved_by: str | None = Field(default=None, description="'policy' | 'vlm' | 'human'")
    pending_human: bool
    calibrated_p: float | None = None
    tier1_decision: str | None = Field(
        default=None, description="the detector-only PASS/FAIL/ESCALATE call, pre-VLM/human")
    applied_target_prevalence: float | None = None
    pi_source: float | None = None
    expected_costs: dict[str, float] | None = None
    indifference_points: dict[str, float] | None = None
    vlm: VLMInfo | None = None
    run_guard_warnings: list[str] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    run_id: str
    category: str
    n: int
    n_good: int
    n_defective: int
    pi_source: float
    image_auroc: float
    default_target_prevalence: float
    locked_cost: dict[str, float]
    realistic_cost: dict[str, float]
    guard_warnings: list[str]
    vlm_provider: str
    vlm_model: str
    auth_env_name: str = Field(description="the env-var NAME holding the API key — never its value")
    auth_enabled: bool


class HealthResponse(BaseModel):
    status: str
    run_id: str
    auth_enabled: bool
