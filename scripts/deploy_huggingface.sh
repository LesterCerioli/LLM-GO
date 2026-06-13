#!/usr/bin/env bash
# scripts/deploy_huggingface.sh — publica o GoLLM no Hugging Face Hub
set -euo pipefail

: "${HF_TOKEN:?Defina HF_TOKEN antes de executar}"
: "${HF_REPO_ID:?Defina HF_REPO_ID (ex: meu-org/llm-go-350m)}"

CKPT_DIR=${CKPT_DIR:-checkpoints/final}
TOK_DIR=${TOK_DIR:-data/tokenizer}
PRIVATE=${PRIVATE:-false}
COMMIT_MSG=${COMMIT_MSG:-"Upload GoLLM checkpoint"}

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "ERRO: Checkpoint não encontrado em $CKPT_DIR."
    exit 1
fi

echo "==> Publicando GoLLM no Hugging Face Hub"
echo "    repo   : $HF_REPO_ID"
echo "    ckpt   : $CKPT_DIR"
echo "    tok    : $TOK_DIR"
echo "    privado: $PRIVATE"
echo ""

llm-go-deploy \
    --ckpt-dir  "$CKPT_DIR" \
    --tok-dir   "$TOK_DIR" \
    --repo-id   "$HF_REPO_ID" \
    --token     "$HF_TOKEN" \
    --message   "$COMMIT_MSG" \
    $([ "$PRIVATE" = "true" ] && echo "--private" || echo "--public")

echo ""
echo "✅ Modelo publicado em: https://huggingface.co/$HF_REPO_ID"
