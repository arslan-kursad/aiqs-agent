"""The FastAPI Intelligent API — ``aiqs-serve``.

    uv run aiqs-serve --run <run_id>                 # mock VLM backend, no API key needed
    uv run aiqs-serve --run <run_id> --provider anthropic --image-root datasets/

Serving is TORCH-FREE by construction: this module and everything it imports touch only
the persisted ``image_scores.csv`` (via ``aiqs.api.artifact``) and the VLM backend seam —
never anomalib/torch. The detector stays an offline producer; this consumes its output
exactly like ``aiqs-decide`` does, plus adjudicates through the SAME LangGraph
``aiqs-graph`` uses (one graph, two front doors).
"""

from __future__ import annotations

import argparse
import base64
import os
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from aiqs.api.artifact import DecisionArtifact, load_artifact
from aiqs.api.schemas import (
    AdjudicateRequest,
    AdjudicateResponse,
    ConfigResponse,
    HealthResponse,
    HumanVerdictRequest,
    VLMInfo,
)
from aiqs.graph.build import build_graph
from aiqs.vlm.backend import MockVLMBackend, VLMBackend
from aiqs.vlm_crop import build_backend as build_vlm_backend

DEFAULT_AUTH_ENV = "AIQS_API_KEY"


class ServeContext:
    """Everything one ``aiqs-serve`` process holds: the artifact, the compiled graph,
    the image-path root (path-traversal guard), and auth config."""

    def __init__(self, artifact: DecisionArtifact, backend: VLMBackend, checkpointer,
                image_root: Path | None, auth_env: str, provider: str, model: str):
        self.artifact = artifact
        self.graph = build_graph(artifact, backend, checkpointer)
        self.image_root = image_root.resolve() if image_root else None
        self.auth_env = auth_env
        self.provider = provider
        self.model = model
        self._upload_dir = artifact.run_dir / "serve_uploads"

    def resolve_image_path(self, image_path: str | None, image_b64: str | None) -> str | None:
        """image_b64 is the safe default path (bytes never touch the server's
        filesystem namespace); image_path is confined to --image-root to close the
        path-traversal hole a raw client-supplied path would otherwise open."""
        if image_b64:
            self._upload_dir.mkdir(parents=True, exist_ok=True)
            try:
                data = base64.b64decode(image_b64, validate=True)
            except Exception as e:
                raise HTTPException(400, f"image_b64 is not valid base64: {e}") from e
            dest = self._upload_dir / f"{uuid.uuid4().hex}.png"
            dest.write_bytes(data)
            return str(dest)
        if image_path is None:
            return None
        if self.image_root is None:
            raise HTTPException(
                400, "image_path serving is disabled on this server (no --image-root "
                    "configured); use image_b64 instead.")
        candidate = (self.image_root / image_path).resolve()
        if candidate != self.image_root and self.image_root not in candidate.parents:
            raise HTTPException(400, "image_path escapes the configured --image-root.")
        if not candidate.is_file():
            raise HTTPException(400, f"image_path not found under --image-root: {image_path!r}")
        return str(candidate)

    def resolve_target_prevalence(self, requested) -> float:
        if requested is None:
            return self.artifact.default_target_prevalence
        if requested == "native":
            return self.artifact.pi_source
        return float(requested)


def _to_response(item_id: str, state: dict[str, Any], artifact: DecisionArtifact,
                 pending: bool | None = None) -> AdjudicateResponse:
    is_pending = pending if pending is not None else ("__interrupt__" in state)
    vlm = None
    if state.get("vlm_verdict") is not None:
        vlm = VLMInfo(fired=True, verdict=state["vlm_verdict"],
                      confidence=state.get("vlm_confidence"),
                      reasoning=state.get("vlm_reasoning"),
                      tokens_in=state.get("tokens_in"), tokens_out=state.get("tokens_out"))
    decision = state.get("final_decision")
    if decision is None and is_pending:
        decision = "pending_human"
    return AdjudicateResponse(
        item_id=item_id, decision=decision, resolved_by=state.get("resolved_by"),
        pending_human=is_pending, calibrated_p=state.get("detector_p"),
        tier1_decision=state.get("tier1_decision"),
        applied_target_prevalence=state.get("target_prevalence"),
        pi_source=state.get("pi_source"), expected_costs=state.get("expected_costs"),
        indifference_points=state.get("indifference_points"), vlm=vlm,
        run_guard_warnings=list(artifact.guard_warnings),
    )


def create_app(ctx: ServeContext) -> FastAPI:
    app = FastAPI(title="AIQS-Agent Intelligent API",
                  description="Auditable, regime-conditional, cost-aware adjudication "
                              "serving — a torch-free layer over an offline detector run.",
                  version="0.1.0")

    def require_auth(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.getenv(ctx.auth_env)
        if not expected:
            return  # dev mode: auth disabled when the env var is unset (see /health, docs)
        if x_api_key != expected:
            raise HTTPException(401, "invalid or missing API key")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", run_id=ctx.artifact.run_id,
                              auth_enabled=bool(os.getenv(ctx.auth_env)))

    @app.get("/config", response_model=ConfigResponse, dependencies=[Depends(require_auth)])
    def get_config() -> ConfigResponse:
        a = ctx.artifact
        return ConfigResponse(
            run_id=a.run_id, category=a.category, n=a.n, n_good=a.n_good,
            n_defective=a.n_defective, pi_source=a.pi_source, image_auroc=a.auroc,
            default_target_prevalence=a.default_target_prevalence,
            locked_cost=_cost_dict(a.locked_cost), realistic_cost=_cost_dict(a.realistic_cost),
            guard_warnings=list(a.guard_warnings), vlm_provider=ctx.provider,
            vlm_model=ctx.model, auth_env_name=ctx.auth_env,
            auth_enabled=bool(os.getenv(ctx.auth_env)),
        )

    @app.post("/adjudicate", response_model=AdjudicateResponse,
             dependencies=[Depends(require_auth)])
    def adjudicate(req: AdjudicateRequest) -> AdjudicateResponse:
        item_id = req.item_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": item_id}}
        existing = ctx.graph.get_state(config)
        if existing.values:
            done = not existing.next
            raise HTTPException(409, detail={
                "message": "item_id already adjudicated" if done
                          else "item_id already pending human review",
                "item_id": item_id,
                "see": (f"GET /decisions/{item_id}" if done
                       else f"POST /human-verdict/{item_id}"),
            })

        image_path = ctx.resolve_image_path(req.image_path, req.image_b64)
        cost = req.cost_matrix
        initial = {
            "item_id": item_id,
            "detector_score": req.anomaly_score,
            "target_prevalence": ctx.resolve_target_prevalence(req.target_prevalence),
            "cost_false_accept": (cost.false_accept if cost and cost.false_accept is not None
                                  else ctx.artifact.locked_cost.false_accept),
            "cost_false_reject": (cost.false_reject if cost and cost.false_reject is not None
                                  else ctx.artifact.locked_cost.false_reject),
            "cost_escalation": (cost.escalation if cost and cost.escalation is not None
                               else ctx.artifact.locked_cost.escalation),
            "image_path": image_path,
            "lam": req.lam,
        }
        try:
            out = ctx.graph.invoke(initial, config=config)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return _to_response(item_id, out, ctx.artifact)

    @app.post("/human-verdict/{item_id}", response_model=AdjudicateResponse,
             dependencies=[Depends(require_auth)])
    def human_verdict(item_id: str, req: HumanVerdictRequest) -> AdjudicateResponse:
        config = {"configurable": {"thread_id": item_id}}
        snap = ctx.graph.get_state(config)
        if not snap.values:
            raise HTTPException(404, f"no adjudication found for item_id={item_id!r}")
        if not snap.next:
            raise HTTPException(409, detail={"message": "item already finalized",
                                             "see": f"GET /decisions/{item_id}"})
        out = ctx.graph.invoke(
            Command(resume={"decision": req.decision, "reviewer": req.reviewer,
                            "note": req.note}),
            config=config)
        return _to_response(item_id, out, ctx.artifact)

    @app.get("/decisions/{item_id}", response_model=AdjudicateResponse,
            dependencies=[Depends(require_auth)])
    def get_decision(item_id: str) -> AdjudicateResponse:
        config = {"configurable": {"thread_id": item_id}}
        snap = ctx.graph.get_state(config)
        if not snap.values:
            raise HTTPException(404, f"no adjudication found for item_id={item_id!r}")
        return _to_response(item_id, snap.values, ctx.artifact, pending=bool(snap.next))

    return app


def _cost_dict(cost) -> dict[str, float]:
    return {"false_accept": cost.false_accept, "false_reject": cost.false_reject,
            "escalation": cost.escalation}


def _build_backend(args) -> VLMBackend:
    if args.provider == "mock":
        return MockVLMBackend(verdict_fn=lambda state, rng: (
            "unsure", 0.5, "mock backend (aiqs-serve --provider mock): no real VLM call"))
    return build_vlm_backend(args.provider, args.model, base_url=args.base_url,
                             api_key_env=args.api_key_env)


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the AIQS-Agent Intelligent API.")
    p.add_argument("--run", help="results/runs/<run_id> (default: latest).")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--provider", default="mock",
                   choices=["mock", "anthropic", "openai_compatible"],
                   help="VLM backend for the ESCALATE path (default: mock — no API key needed).")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--base-url", default=None, help="required for --provider openai_compatible")
    p.add_argument("--api-key-env", default=None,
                   help="env var NAME holding the VLM provider's key (openai_compatible only)")
    p.add_argument("--image-root", default=None,
                   help="directory image_path requests are confined to (path-traversal guard). "
                        "Unset => image_path requests are rejected; use image_b64.")
    p.add_argument("--auth-env", default=DEFAULT_AUTH_ENV,
                   help=f"env var NAME holding this API's own auth key (default {DEFAULT_AUTH_ENV}). "
                        "Unset in the environment => auth is disabled (dev mode).")
    p.add_argument("--checkpoint-db", default=None,
                   help="sqlite path (default: <run_dir>/graph_checkpoints.sqlite)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    artifact = load_artifact(results_dir, args.run)
    backend = _build_backend(args)
    ckpt_path = Path(args.checkpoint_db) if args.checkpoint_db else \
        artifact.run_dir / "graph_checkpoints.sqlite"

    if not os.getenv(args.auth_env):
        print(f"[aiqs-serve] WARNING: {args.auth_env} is not set — API auth is DISABLED "
              "(dev mode). Set it before exposing this server beyond localhost.")

    with SqliteSaver.from_conn_string(str(ckpt_path)) as checkpointer:
        ctx = ServeContext(artifact, backend, checkpointer,
                          Path(args.image_root) if args.image_root else None,
                          args.auth_env, args.provider, args.model)
        app = create_app(ctx)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
