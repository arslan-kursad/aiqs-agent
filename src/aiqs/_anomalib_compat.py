"""Detect the installed anomalib major version.

The detector/data seams (``detector.py`` / ``data.py``) dispatch on this so the SAME
public API works against two stacks that never coexist:

  * **anomalib 1.2** — the pinned local Intel-mac stack (`_backend_v1`, in-line in the
    seams). Verified by `make test` / `make smoke`.
  * **anomalib 2.x** — a GPU-host-only optional extra (`_detector_v2.py` / `_data_v2.py`).
    Imported ONLY when anomalib 2.x is installed; never exercised on the local 1.2 stack
    (those modules are GPU-host-verified, see CLAUDE.md Phase-2B Stage 1).

Keeping this a one-liner module (not a class) avoids any import-time anomalib touch — the
seams import this, then lazily import the version-appropriate backend inside the call.
"""

from __future__ import annotations


def anomalib_major() -> int:
    """Major version of the installed anomalib (1 for the pinned 1.2 stack, 2 for AD2)."""
    import anomalib

    return int(str(anomalib.__version__).split(".", 1)[0])
