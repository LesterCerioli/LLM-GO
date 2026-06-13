.PHONY: install install-gpu collect tokenize train evaluate generate deploy test lint fmt

PYTHON     := python3.12
PIP        := $(PYTHON) -m pip
MODEL_SIZE ?= medium
CKPT_DIR   ?= checkpoints/final
TOK_DIR    ?= data/tokenizer
HF_REPO    ?= your-org/llm-go-$(MODEL_SIZE)

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

install-gpu:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-gpu.txt
	$(PIP) install -e ".[dev]"

# ── Data pipeline ──────────────────────────────────────────────────────────
collect:
	llm-go-collect --token $$GITHUB_TOKEN --max-repos 50000

tokenize:
	llm-go-tokenize --raw-dir data/raw --out-dir $(TOK_DIR) --vocab-size 32000

preprocess:
	$(PYTHON) -c "\
from llm_go.tokenizer import GoTokenizer; \
from llm_go.data import GoPreprocessor; \
tok = GoTokenizer.load('$(TOK_DIR)'); \
p = GoPreprocessor(tok); \
p.run()"

# ── Training ───────────────────────────────────────────────────────────────
train:
	llm-go-train --model-size $(MODEL_SIZE) --data-dir data/processed \
	             --ckpt-dir checkpoints --log-dir logs

train-small:
	$(MAKE) train MODEL_SIZE=small

train-medium:
	$(MAKE) train MODEL_SIZE=medium

train-large:
	$(MAKE) train MODEL_SIZE=large

# ── Evaluation ─────────────────────────────────────────────────────────────
evaluate:
	llm-go-evaluate --model-dir $(CKPT_DIR) --tok-dir $(TOK_DIR)

# ── Generation ─────────────────────────────────────────────────────────────
generate:
	llm-go-generate --model-dir $(CKPT_DIR) --tok-dir $(TOK_DIR)

# ── Deploy ─────────────────────────────────────────────────────────────────
deploy:
	llm-go-deploy --ckpt-dir $(CKPT_DIR) --tok-dir $(TOK_DIR) \
	              --repo-id $(HF_REPO) --token $$HF_TOKEN

# ── Tensorboard ────────────────────────────────────────────────────────────
tb:
	tensorboard --logdir logs

# ── Quality ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=llm_go --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

fmt:
	black src/ tests/
	ruff check --fix src/ tests/

# ── Full pipeline ──────────────────────────────────────────────────────────
pipeline: collect tokenize preprocess train evaluate deploy
