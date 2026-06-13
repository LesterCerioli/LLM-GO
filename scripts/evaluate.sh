#!/usr/bin/env bash
# scripts/evaluate.sh — avalia um checkpoint em perplexidade, pass@k e gofmt
set -euo pipefail

CKPT_DIR=${CKPT_DIR:-checkpoints/final}
TOK_DIR=${TOK_DIR:-data/tokenizer}
DATA_DIR=${DATA_DIR:-data/processed}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_BATCHES=${MAX_BATCHES:-200}

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "ERRO: Checkpoint não encontrado em $CKPT_DIR."
    echo "      Execute scripts/train.sh primeiro."
    exit 1
fi

echo "==> Avaliando checkpoint: $CKPT_DIR"
echo "    tok=$TOK_DIR  data=$DATA_DIR"
echo ""

llm-go-evaluate \
    --model-dir   "$CKPT_DIR" \
    --tok-dir     "$TOK_DIR" \
    --data-dir    "$DATA_DIR" \
    --batch-size  "$BATCH_SIZE" \
    --max-batches "$MAX_BATCHES"
