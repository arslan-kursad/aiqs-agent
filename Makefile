# AIQS-Agent — Phase 0 task runner.
# Usage:
#   make install
#   make train                      # full config budget
#   make smoke                      # tiny end-to-end sanity run
#   make baseline CATEGORY=screw    # train + eval
#   make eval

CONFIG   ?= configs/default.yaml
CATEGORY ?= screw
RUN      := uv run

.PHONY: help install data train eval baseline smoke clean

help:
	@echo "Targets: install | data | train | eval | baseline | smoke | clean"
	@echo "Vars:    CONFIG=$(CONFIG)  CATEGORY=$(CATEGORY)"

install:
	uv sync

# Fetch one MVTec AD category into anomalib's layout (works around anomalib
# 1.2.0's dead download URL — see src/aiqs/prepare_data.py). Idempotent.
data:
	$(RUN) aiqs-prepare-data --category $(CATEGORY)

# Train the detector on one category (ensures data is present first).
train: data
	$(RUN) aiqs-train --config $(CONFIG) --category $(CATEGORY)

# Evaluate the latest trained checkpoint and persist metrics to results/.
eval:
	$(RUN) aiqs-eval --config $(CONFIG) --category $(CATEGORY)

# Full baseline: train then evaluate.
baseline: train eval

# Fast end-to-end smoke test (a handful of steps) to verify the pipeline wiring
# before committing to a multi-hour CPU training run.
SMOKE_IMAGENETTE := datasets/_imagenette_smoke

smoke: data
	$(RUN) aiqs-prepare-data --make-synthetic-imagenette $(SMOKE_IMAGENETTE)
	$(RUN) aiqs-train --config $(CONFIG) --category $(CATEGORY) \
		--max-steps 10 --imagenet-dir $(SMOKE_IMAGENETTE)
	$(RUN) aiqs-eval  --config $(CONFIG) --category $(CATEGORY)

clean:
	rm -rf models/ lightning_logs/
	@echo "Removed model checkpoints and lightning logs (datasets/ and results/ kept)."
