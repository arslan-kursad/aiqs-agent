"""Shared served-model verification — the silent-downgrade guard every VLM backend uses.

An alias may legitimately resolve to its dated full id (e.g. ``claude-haiku-4-5`` ->
``claude-haiku-4-5-20251001``) — that is the SAME model, not a downgrade. A different
model family/version still fails. Kept strict deliberately: the guard's value IS its
strictness — a provider-specific model-string quirk should be handled by passing the
provider's own served-string convention through ``requested``/config, never by loosening
this matcher globally (see CLAUDE.md decision log).
"""

from __future__ import annotations


def model_matches(requested: str, served: str | None) -> bool:
    """True if ``served`` is exactly the requested model or its dated/suffixed full id."""
    if served is None:
        return False
    return served == requested or served.startswith(requested + "-")


def assert_model_matches(requested: str, served: str | None) -> None:
    """Raise RuntimeError (loud, never silent) if ``served`` is not the requested model."""
    if not model_matches(requested, served):
        raise RuntimeError(
            f"SERVED MODEL MISMATCH: expected {requested!r}, API served {served!r}. "
            "Stopping — results on a substitute model are not valid evidence.")
