#!/usr/bin/env bash
# scripts/train.sh — executa o treinamento do GoLLM
set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-medium}       # small | medium | large | xl
DATA_DIR=${DATA_DIR:-data/processed}
CKPT_DIR=${CKPT_DIR:-checkpoints}
LOG_DIR=${LOG_DIR:-logs}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-100000}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
GRAD_ACCUM=${GRAD_ACCUM:-4}
PRECISION=${PRECISION:-bfloat16}       # float32 | float16 | bfloat16
GPUS=${GPUS:--1}                       # -1 = todos, 0 = GPU 0, 1 = GPU 1...

# Verifica dados pré-processados
TRAIN_SHARDS=$(find "$DATA_DIR/train" -name "*.tfrecord" 2>/dev/null | wc -l)
if [ "$TRAIN_SHARDS" -eq 0 ]; then
    echo "ERRO: Nenhum TFRecord encontrado em $DATA_DIR/train."
    echo "      Execute scripts/preprocess.sh primeiro."
    exit 1
fi

echo "==> Iniciando treinamento GoLLM"
echo "    model_size=$MODEL_SIZE  batch=$BATCH_SIZE  steps=$MAX_STEPS"
echo "    precision=$PRECISION  gpus=$GPUS  grad_accum=$GRAD_ACCUM"
echo "    train_shards=$TRAIN_SHARDS"
echo ""

# XLA JIT compilation (aceleração de CPU/GPU)
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TF_ENABLE_ONEDNN_OPTS=0

llm-go-train \
    --model-size   "$MODEL_SIZE" \
    --data-dir     "$DATA_DIR" \
    --ckpt-dir     "$CKPT_DIR" \
    --log-dir      "$LOG_DIR" \
    --batch-size   "$BATCH_SIZE" \
    --max-steps    "$MAX_STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --grad-accum   "$GRAD_ACCUM" \
    --precision    "$PRECISION" \
    --gpus         "$GPUS"

echo ""
echo "✅ Treinamento concluído. Checkpoints em $CKPT_DIR"
