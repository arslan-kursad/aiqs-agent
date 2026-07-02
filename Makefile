# AIQS-Agent — task runner.
# Usage:
#   make install
#   make train                      # full config budget
#   make smoke                      # tiny end-to-end sanity run
#   make baseline CATEGORY=screw    # train + eval
#   make eval
#   make decide                     # Phase-1 adjudication on the latest run
#   make decide RUN=<run_id>        # ...on a specific results/runs/<run_id>
#   make test                       # unit tests

CONFIG   ?= configs/patchcore_cpu.yaml
CATEGORY ?= screw
RUN      ?=                 # Phase-1 run id (empty => latest run)
UV       := uv run

.PHONY: help install data train eval baseline smoke smoke-patchcore decide sim test clean

help:
	@echo "Targets: install | data | train | eval | baseline | smoke | decide | test | clean"
	@echo "Vars:    CONFIG=$(CONFIG)  CATEGORY=$(CATEGORY)  RUN=$(RUN)"

install:
	uv sync

# Fetch one MVTec AD category into anomalib's layout (works around anomalib
# 1.2.0's dead download URL — see src/aiqs/prepare_data.py). Idempotent.
data:
	$(UV) aiqs-prepare-data --category $(CATEGORY)

# Train the detector on one category (ensures data is present first).
train: data
	$(UV) aiqs-train --config $(CONFIG) --category $(CATEGORY)

# Evaluate the latest trained checkpoint and persist metrics to results/.
eval:
	$(UV) aiqs-eval --config $(CONFIG) --category $(CATEGORY)

# Full baseline: train then evaluate.
baseline: train eval

# Fast end-to-end smoke test (a handful of steps) to verify the pipeline wiring
# before committing to a multi-hour CPU training run.
SMOKE_IMAGENETTE := datasets/_imagenette_smoke

smoke: data
	$(UV) aiqs-prepare-data --make-synthetic-imagenette $(SMOKE_IMAGENETTE)
	$(UV) aiqs-train --config $(CONFIG) --category $(CATEGORY) \
		--max-steps 10 --imagenet-dir $(SMOKE_IMAGENETTE)
	$(UV) aiqs-eval  --config $(CONFIG) --category $(CATEGORY)

# PatchCore smoke: no --imagenet-dir needed, no --max-steps (single epoch).
smoke-patchcore: data
	$(UV) aiqs-train --config configs/patchcore_cpu.yaml --category $(CATEGORY)
	$(UV) aiqs-eval  --config configs/patchcore_cpu.yaml --category $(CATEGORY)

# Phase-1 decision layer: calibrate + cost-matrix PASS/FAIL/ESCALATE on persisted
# per-image scores. RUN defaults to the latest run.
decide:
	$(UV) aiqs-decide $(if $(RUN),--run $(RUN),)

# SYNTHETIC machinery validation (NOT real-data evidence) — proves the decision
# code is correct on a detector that actually separates. Walled off under
# results/synthetic_validation/.
sim:
	$(UV) aiqs-sim-decision

# Phase-2A VLM second-look on the ESCALATE bucket. RUN defaults to the latest run.
# MOCK=1 runs the wiring smoke (no API key, no cost); otherwise needs ANTHROPIC_API_KEY.
vlm:
	$(UV) aiqs-vlm $(if $(RUN),--run $(RUN),) $(if $(MOCK),--mock,)

# Phase-2B Stage-3 two-arm full-vs-crop experiment (needs the run's anomaly maps).
vlm-crop:
	$(UV) aiqs-vlm-crop $(if $(RUN),--run $(RUN),) $(if $(MOCK),--mock,)

# Unit tests for the decision policy / calibration / guard (dev group).
test:
	uv run --group dev pytest -q

clean:
	rm -rf models/ lightning_logs/
	@echo "Removed model checkpoints and lightning logs (datasets/ and results/ kept)."
