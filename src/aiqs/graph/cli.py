"""Thin CLI to run one item through the LangGraph adjudication graph, for demo/debug —
NOT the serving path (that is ``aiqs-serve``, which shares this same ``build_graph``).

    uv run aiqs-graph --run <run_id> --score 0.83
    uv run aiqs-graph --run <run_id> --score 0.83 --image-path some.png
    uv run aiqs-graph --run <run_id> --item-id demo-1 --resume-decision fail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from aiqs.api.artifact import load_artifact
from aiqs.graph.build import build_graph
from aiqs.vlm.backend import MockVLMBackend


def _default_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "graph_checkpoints.sqlite"


def main() -> None:
    p = argparse.ArgumentParser(description="Run one item through the adjudication graph.")
    p.add_argument("--run", help="results/runs/<run_id> (default: latest).")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--item-id", default="cli-demo")
    p.add_argument("--score", type=float, help="raw detector anomaly score.")
    p.add_argument("--target-prevalence", type=float, default=None,
                   help="default: the artifact's default (0.02).")
    p.add_argument("--image-path", default=None)
    p.add_argument("--resume-decision", choices=["pass", "fail"], default=None,
                   help="resume a paused (item-id) with this human verdict.")
    p.add_argument("--reviewer", default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    artifact = load_artifact(results_dir, args.run)
    backend = MockVLMBackend(verdict_fn=lambda state, rng: ("unsure", 0.5, "cli demo: mock backend, no API call"))

    ckpt_path = _default_checkpoint_path(artifact.run_dir)
    with SqliteSaver.from_conn_string(str(ckpt_path)) as saver:
        graph = build_graph(artifact, backend, saver)
        config = {"configurable": {"thread_id": args.item_id}}

        if args.resume_decision is not None:
            out = graph.invoke(
                Command(resume={"decision": args.resume_decision, "reviewer": args.reviewer}),
                config=config)
        else:
            if args.score is None:
                p.error("--score is required unless --resume-decision is given.")
            initial = {
                "item_id": args.item_id,
                "detector_score": args.score,
                "target_prevalence": (args.target_prevalence
                                      if args.target_prevalence is not None
                                      else artifact.default_target_prevalence),
                "cost_false_accept": artifact.locked_cost.false_accept,
                "cost_false_reject": artifact.locked_cost.false_reject,
                "cost_escalation": artifact.locked_cost.escalation,
                "image_path": args.image_path,
            }
            out = graph.invoke(initial, config=config)

        print(json.dumps(out, indent=2, default=str))
        if "__interrupt__" in out:
            print(f"\n-> PAUSED for human review. Resume with:\n"
                  f"   uv run aiqs-graph --run {artifact.run_dir.name} "
                  f"--item-id {args.item_id} --resume-decision pass|fail\n")


if __name__ == "__main__":
    main()
