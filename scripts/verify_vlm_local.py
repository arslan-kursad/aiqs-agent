#!/usr/bin/env python3
"""Phase-2B Stage 0.1 + 0.2 local pre-flight — NOT a headline, a hygiene check.

Makes a few REAL Anthropic API calls (a few cents) to answer two questions before the
AD2 migration, locally, without Kaggle/Langfuse:

  0.1  Which model does the API actually serve? LOG ``msg.model`` and STOP LOUDLY if it
       is not ``claude-sonnet-4-6`` (no silent downgrade — the "fork quietly dropped to
       3.5" lesson). A wrong model invalidates every 2A finding.

  0.2  How many tokens does one second-look cost? Read ``msg.usage`` (returned on every
       response — Langfuse not required) over a handful of calls, then project an AD2
       budget = per-call x expected ESCALATE-bucket size.

Reuses the production ``AnthropicVLMBackend`` content construction (no fork). Loads
``ANTHROPIC_API_KEY`` from a local ``.env`` (gitignored).

Honest caveat (0.2): capsule images are Colab-only on this host (image_scores paths are
``/content/...``), so per-call cost is measured on LOCAL screw images. The cost is
dominated by the fixed ~512px image block + the fixed prompt, so it is category-
independent and transfers as a representative estimate. The crop adds a SECOND image
block in Stage 1; that increment is measured then, not here.

Run: ``uv run python scripts/verify_vlm_local.py [N]``  (N = #calls, default 8)
The durable record of the result goes into the CLAUDE.md decision log.
"""

from __future__ import annotations

import os
import statistics as stats
import sys
from pathlib import Path

EXPECTED_MODEL = "claude-sonnet-4-6"
SCREW_TEST = Path("datasets/mvtec/screw/test")
# Illustrative AD2 ESCALATE-bucket sizes to project the budget over (the Stage-2 power
# gate wants ESCALATE∩good AND n_dw each >= ~30, so a powered bucket is ~60-80 items).
PROJECT_BUCKETS = (40, 80)


def _load_env() -> None:
    """Pull ANTHROPIC_API_KEY from a local .env if python-dotenv is present (it ships in
    the [vlm] extra); otherwise rely on an already-exported env var."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _pick_screw_images(n: int) -> list[str]:
    """A representative spread across the screw test/ defect folders + good."""
    imgs = sorted(str(p) for p in SCREW_TEST.rglob("*.png"))
    if not imgs:
        sys.exit(f"STOP: no local screw test images under {SCREW_TEST} "
                 "(run `make data` for the screw category first).")
    step = max(1, len(imgs) // n)
    return imgs[::step][:n]


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 8

    _load_env()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("STOP: ANTHROPIC_API_KEY not found. Put it in a local .env "
                 "(ANTHROPIC_API_KEY=sk-ant-...) or export it, then re-run.")

    # Lazy imports so --help / a missing key fail fast without loading the SDK.
    from aiqs.vlm.backend import MODEL, AnthropicVLMBackend
    from aiqs.vlm.prompt import SYSTEM_PROMPT
    from aiqs.vlm.state import VLMState

    backend = AnthropicVLMBackend()           # api_key=None => SDK reads ANTHROPIC_API_KEY
    client = backend._client_lazy()           # reuse the real client construction
    paths = _pick_screw_images(n)

    print(f"== Stage 0.1/0.2 local pre-flight ==  ({len(paths)} real calls, model const "
          f"MODEL={MODEL!r})\n")

    served: set[str] = set()
    in_toks: list[int] = []
    out_toks: list[int] = []
    for i, p in enumerate(paths, 1):
        # detector_score/detector_p are unused by full-image content construction; dummy.
        state = VLMState(image_path=p, detector_score=0.5, detector_p=0.5)
        content = backend._build_content(state)   # reuse production prompt+image (no fork)
        msg = client.messages.create(
            model=backend.model, max_tokens=backend.max_tokens,
            temperature=backend.temperature,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        )
        served.add(msg.model)
        u = msg.usage
        in_toks.append(u.input_tokens)
        out_toks.append(u.output_tokens)
        print(f"[{i}/{len(paths)}] {Path(p).name:>16}  served={msg.model}  "
              f"in={u.input_tokens:>5}  out={u.output_tokens:>4}  stop={msg.stop_reason}")

    # ---- 0.1 gate: served model must be exactly the expected string -------------------
    print()
    if served != {EXPECTED_MODEL}:
        print(f"*** STOP (0.1): served model(s) {sorted(served)} != expected "
              f"{EXPECTED_MODEL!r}. ***")
        print("Do NOT trust any 2A/2B VLM result until this matches "
              "(the silent 3.5-downgrade lesson). Investigate access/auth, not the string.")
        return 2
    print(f"0.1 OK — every call served {EXPECTED_MODEL!r} (matches code MODEL const).")

    # ---- 0.2 budget: per-call usage + AD2 projection ----------------------------------
    in_mean, out_mean = stats.mean(in_toks), stats.mean(out_toks)
    print(f"\n0.2 token usage over {len(paths)} calls (full-image only, ~512px):")
    print(f"  input : mean {in_mean:7.1f}  min {min(in_toks)}  max {max(in_toks)}")
    print(f"  output: mean {out_mean:7.1f}  min {min(out_toks)}  max {max(out_toks)}")
    print(f"  per-call total ~ {in_mean + out_mean:7.1f} tokens")
    print("  AD2 bucket projection (full-image-only; crop adds a 2nd image block, Stage 1):")
    for b in PROJECT_BUCKETS:
        print(f"    bucket={b:>3}:  ~{(in_mean) * b:>9.0f} in  +  ~{(out_mean) * b:>7.0f} out")
    print("\nNOTE: measured on LOCAL screw images (capsule is Colab-only here); per-call "
          "cost is image-size + prompt driven, so category-independent. Record these "
          "numbers in the CLAUDE.md decision log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
