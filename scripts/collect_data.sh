#!/usr/bin/env bash
# scripts/collect_data.sh — coleta código Go do GitHub e da stdlib
set -euo pipefail

: "${GITHUB_TOKEN:?Defina GITHUB_TOKEN antes de executar}"

RAW_DIR=${RAW_DIR:-data/raw}
MAX_REPOS=${MAX_REPOS:-50000}
MIN_STARS=${MIN_STARS:-10}
GOROOT=${GOROOT:-$(go env GOROOT 2>/dev/null || echo "")}

echo "==> Iniciando coleta de repositórios Go (max=$MAX_REPOS, min_stars=$MIN_STARS)..."
llm-go-collect \
    --token       "$GITHUB_TOKEN" \
    --out-dir     "$RAW_DIR" \
    --max-repos   "$MAX_REPOS" \
    --min-stars   "$MIN_STARS"

echo "==> Coletando stdlib Go..."
if [ -n "$GOROOT" ] && [ -d "$GOROOT" ]; then
    echo "    Usando GOROOT local: $GOROOT"
    llm-go-collect --token "$GITHUB_TOKEN" --stdlib --go-root "$GOROOT"
else
    echo "    GOROOT não encontrado, baixando do GitHub..."
    llm-go-collect --token "$GITHUB_TOKEN" --stdlib
fi

echo ""
FILES=$(find "$RAW_DIR" -name "*.go" 2>/dev/null | wc -l)
echo "✅ Coleta concluída: $FILES arquivos .go em $RAW_DIR"
