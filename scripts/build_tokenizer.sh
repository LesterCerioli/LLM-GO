#!/usr/bin/env bash
# scripts/build_tokenizer.sh — treina o tokenizador BPE no corpus Go coletado
set -euo pipefail

RAW_DIR=${RAW_DIR:-data/raw}
TOK_DIR=${TOK_DIR:-data/tokenizer}
VOCAB_SIZE=${VOCAB_SIZE:-32000}

FILES=$(find "$RAW_DIR" -name "*.go" 2>/dev/null | wc -l)
if [ "$FILES" -eq 0 ]; then
    echo "ERRO: Nenhum arquivo .go encontrado em $RAW_DIR."
    echo "      Execute scripts/collect_data.sh primeiro."
    exit 1
fi

echo "==> Treinando tokenizador BPE em $FILES arquivos Go..."
echo "    vocab_size=$VOCAB_SIZE  saída=$TOK_DIR"

llm-go-tokenize \
    --raw-dir   "$RAW_DIR" \
    --out-dir   "$TOK_DIR" \
    --vocab-size "$VOCAB_SIZE"

echo ""
echo "✅ Tokenizador salvo em $TOK_DIR"
echo "   Arquivos:"
ls -lh "$TOK_DIR"
