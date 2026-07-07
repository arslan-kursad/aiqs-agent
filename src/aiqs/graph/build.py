"""Assemble the compiled LangGraph: ingest -> calibrate -> cost_policy ->
[PASS/FAIL: finalize] | [ESCALATE: vlm_second_look -> vlm_abstain_rule -> finalize |
human_interrupt] | [ESCALATE, no image: human_interrupt] -> finalize.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aiqs.api.artifact import DecisionArtifact
from aiqs.graph.state import AdjudicationState
from aiqs.graph.nodes import (
    cost_policy,
    finalize,
    human_interrupt,
    ingest,
    make_calibrate_node,
    make_vlm_second_look_node,
    route_after_cost_policy,
    route_after_vlm_abstain,
    vlm_abstain_rule,
)
from aiqs.vlm.backend import VLMBackend


def build_graph(artifact: DecisionArtifact, backend: VLMBackend, checkpointer
                ) -> CompiledStateGraph:
    g = StateGraph(AdjudicationState)
    g.add_node("ingest", ingest)
    g.add_node("calibrate", make_calibrate_node(artifact))
    g.add_node("cost_policy", cost_policy)
    g.add_node("vlm_second_look", make_vlm_second_look_node(backend))
    g.add_node("vlm_abstain_rule", vlm_abstain_rule)
    g.add_node("human_interrupt", human_interrupt)
    g.add_node("finalize", finalize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "calibrate")
    g.add_edge("calibrate", "cost_policy")
    g.add_conditional_edges("cost_policy", route_after_cost_policy,
                            {"finalize": "finalize", "vlm_second_look": "vlm_second_look",
                             "human_interrupt": "human_interrupt"})
    g.add_edge("vlm_second_look", "vlm_abstain_rule")
    g.add_conditional_edges("vlm_abstain_rule", route_after_vlm_abstain,
                            {"finalize": "finalize", "human_interrupt": "human_interrupt"})
    g.add_edge("human_interrupt", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
