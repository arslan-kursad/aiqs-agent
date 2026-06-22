"""Evaluation backbone — the spine every later phase is measured against.

Phase 0:  detection metrics (metrics.py) + persistence (results.py).
Phase 1+: decision metrics (decision.py) layered on the persisted per-image scores.
"""

from aiqs.eval.results import EvalResult, persist, print_baseline_summary

__all__ = ["EvalResult", "persist", "print_baseline_summary"]
