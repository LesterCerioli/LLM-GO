#!/usr/bin/env bash
# scripts/setup_env.sh — configura o ambiente Python 3.12 e instala dependências
set -euo pipefail

PYTHON=${PYTHON:-python3.12}
VENV_DIR=${VENV_DIR:-.venv}

echo "==> Verificando Python 3.12..."
if ! $PYTHON --version 2>&1 | grep -q "3\.12"; then
    echo "ERRO: Python 3.12 não encontrado. Instale via pyenv ou sistema."
    exit 1
fi

echo "==> Criando virtualenv em $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Atualizando pip..."
pip install --upgrade pip

echo "==> Instalando dependências (CPU)..."
pip install -r requirements.txt

echo "==> Instalando o pacote em modo editable..."
pip install -e ".[dev]"

echo ""
echo "✅ Ambiente pronto. Para ativar: source $VENV_DIR/bin/activate"
echo "   GPU: pip install -r requirements-gpu.txt"
