#!/usr/bin/env bash
set -euo pipefail

# Edit DATA_ROOT and DATASET before running.
DATA_ROOT="/path/to/data/xmno"  
DATASET="xmno"  
MODEL_VARIANT="O"
MESSAGE_PASSING="reference_gated"
CKPT_DIR="./checkpoints/${DATASET}_${MODEL_VARIANT}_${MESSAGE_PASSING}"
LOG_TXT="${CKPT_DIR}/validation_log.txt"

python -m train.train_lacenet \
  --data_root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --model_variant "${MODEL_VARIANT}" \
  --message_passing "${MESSAGE_PASSING}" \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 800 \
  --steps_per_epoch 800 \
  --early_stop_epoch 10 \
  --max_norm 50 \
  --checkpoint_dir "${CKPT_DIR}" \
  --log_txt "${LOG_TXT}" \
  --device cuda:0 \
  --save_model

echo "Best checkpoint: ${CKPT_DIR}/model.pt"
echo "Latest checkpoint: ${CKPT_DIR}/last.pt"
echo "Validation log: ${LOG_TXT}"
