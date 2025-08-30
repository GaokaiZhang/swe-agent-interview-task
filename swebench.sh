#!/bin/bash
set -e
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench \
  --predictions_path predictions/predictions.jsonl \
  --max_workers 1 \
  --run_id full10_run \
  --timeout 1200