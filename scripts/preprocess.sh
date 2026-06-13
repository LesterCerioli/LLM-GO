#!/usr/bin/env bash
# scripts/preprocess.sh — filtra, desduplicamenta, tokeniza e empacota em TFRecords
set -euo pipefail

RAW_DIR=${RAW_DIR:-data/raw}
TOK_DIR=${TOK_DIR:-data/tokenizer}
OUT_DIR=${OUT_DIR:-data/processed}
SEQ_LEN=${SEQ_LEN:-2048}
SHARD_SIZE=${SHARD_SIZE:-10000}

if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
    echo "ERRO: Tokenizador não encontrado em $TOK_DIR."
    echo "      Execute scripts/build_tokenizer.sh primeiro."
    exit 1
fi

echo "==> Pré-processando corpus..."
echo "    raw=$RAW_DIR  tok=$TOK_DIR  out=$OUT_DIR"
echo "    seq_len=$SEQ_LEN  shard_size=$SHARD_SIZE"

python3 -c "
from llm_go.tokenizer import GoTokenizer
from llm_go.data import GoPreprocessor

tok = GoTokenizer.load('$TOK_DIR')
p   = GoPreprocessor(
    tokenizer=$tok,
    raw_dir='$RAW_DIR',
    out_dir='$OUT_DIR',
    seq_len=$SEQ_LEN,
    shard_size=$SHARD_SIZE,
)
counts = p.run()
print('Splits:', counts)
"

echo ""
echo "✅ TFRecords escritos em $OUT_DIR"
for split in train val test; do
    n=$(find "$OUT_DIR/$split" -name "*.tfrecord" 2>/dev/null | wc -l)
    echo "   $split: $n shards"
done
