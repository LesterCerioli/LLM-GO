#!/usr/bin/env bash
# scripts/generate.sh — geração interativa de código Go com o modelo treinado
set -euo pipefail

CKPT_DIR=${CKPT_DIR:-checkpoints/final}
TOK_DIR=${TOK_DIR:-data/tokenizer}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-50}
PROMPT=${1:-}   # prompt opcional como primeiro argumento

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "ERRO: Checkpoint não encontrado em $CKPT_DIR."
    echo "      Execute scripts/train.sh primeiro."
    exit 1
fi

ARGS=(
    --model-dir  "$CKPT_DIR"
    --tok-dir    "$TOK_DIR"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p      "$TOP_P"
    --top-k      "$TOP_K"
)

if [ -n "$PROMPT" ]; then
    ARGS+=(--prompt "$PROMPT")
fi

llm-go-generate "${ARGS[@]}"
