#!/usr/bin/env bash
set -euo pipefail

ASSIGN_DIR="$HOME/swe/swe-agent-interview-task"
PRED_FILE="$ASSIGN_DIR/predictions/predictions.jsonl"

cd ~/swe/SWE-bench

python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench \
  --predictions_path "$PRED_FILE" \
  --max_workers 1 \
  --timeout 3600 \
  --run_id agent-eval-10

echo "[✓] Results saved to ${PRED_FILE}.agent-eval-10.json"

