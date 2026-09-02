#!/usr/bin/env bash
set -euo pipefail

# Edit DATA_ROOT, DATASET, and MODEL_PATH before running.
DATA_ROOT="/path/to/data/xmno" 
DATASET="xmno"
MODEL_VARIANT="O"
MESSAGE_PASSING="reference_gated"
MODEL_PATH="./checkpoints/${DATASET}_${MODEL_VARIANT}_${MESSAGE_PASSING}/model.pt"
TEST_SUBDIR="test"
RESULTS_DIR="./results"

python -m test.test_lacenet \
  --dataset "${DATASET}" \
  --data_root "${DATA_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --test_subdir "${TEST_SUBDIR}" \
  --model_variant "${MODEL_VARIANT}" \
  --message_passing "${MESSAGE_PASSING}" \
  --batch_size 64 \
  --num_workers 4 \
  --results_dir "${RESULTS_DIR}" \
  --device cuda:0

echo "Test result CSV: ${RESULTS_DIR}/${DATASET}_LACENet.csv"
