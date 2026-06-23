# MultiSeedGen - devops entry point.
#
# Quick start (clone & run, no GPU, no Node needed):
#   make run         # install web + CPU rembg into ./.venv and launch the web UI
#   make help        # list every target, grouped
#
# Design notes:
#   * Targets invoke the resolved interpreter directly, so you NEVER need to
#     `source .venv/bin/activate`. By default a local ./.venv is created on first
#     use; if a venv/conda env is already ACTIVE, we install into that instead.
#   * Installs are tiered (lean core -> web -> gpu/sam) and idempotent: stamp files
#     mean `make` only re-downloads when pyproject.toml / package-lock.json change.
#   * GPU is always OPTIONAL. The bare path (`make install`) stays lean.

.DEFAULT_GOAL := help
SHELL := bash
MAKEFLAGS += --no-print-directory

# --------------------------------------------------------------------------- #
# Configuration (override on the command line, e.g. `make run PORT=9000`)      #
# --------------------------------------------------------------------------- #
PYTHON ?= python3
VENV   ?= .venv
HOST   ?= 127.0.0.1
PORT   ?= 8000
CONFIG ?= configs/soybean_healthy.yaml
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu   # set .../cu126 for GPU SAM
SAM_CKPT    := models/sam_vit_b_01ec64.pth

COMPOSE_CPU := docker/compose.yaml
COMPOSE_GPU := docker/compose.gpu.yaml

# Respect an already-active venv/conda env; otherwise manage ./.venv ourselves.
ifdef VIRTUAL_ENV
  PY_BIN := $(VIRTUAL_ENV)/bin/python
else ifdef CONDA_PREFIX
  PY_BIN := $(CONDA_PREFIX)/bin/python
else
  PY_BIN := $(VENV)/bin/python
endif
PY  := $(PY_BIN)
PIP := $(PY_BIN) -m pip
# Env root that owns $(PY_BIN); stamps live here so each env tracks its own state.
ENV_DIR := $(patsubst %/bin/python,%,$(PY_BIN))

# --------------------------------------------------------------------------- #
# Internal rules (file targets — not shown in `help`)                          #
# --------------------------------------------------------------------------- #
# Create ./.venv on demand (only fires when no venv/conda env is active).
$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip

# Stamp: web + CPU rembg. Re-runs only when pyproject.toml changes.
$(ENV_DIR)/.stamp-web: pyproject.toml | $(PY_BIN)
	$(PIP) install -e ".[web,rembg]"
	@touch $@

# Stamp: dev tooling (mirrors CI's `.[dev,web]`).
$(ENV_DIR)/.stamp-dev: pyproject.toml | $(PY_BIN)
	$(PIP) install -e ".[dev,web]"
	@touch $@

# Stamp: frontend node deps. Re-runs only when the lockfile / manifest changes.
frontend/node_modules/.stamp: frontend/package-lock.json frontend/package.json
	cd frontend && npm ci
	@touch $@

# Download the SAM vit_b checkpoint (~358 MB) only if it is not already present.
$(SAM_CKPT):
	mkdir -p $(dir $@)
	curl -fL -o $@ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# --------------------------------------------------------------------------- #
##@ Setup / install (lean -> heavy; only downloads what you ask for)
# --------------------------------------------------------------------------- #
.PHONY: venv install install-web install-rembg install-gpu install-sam install-sam-gpu install-dev download-sam
venv: $(PY_BIN) ## Create ./.venv (skipped if a venv/conda env is active)

install: | $(PY_BIN) ## Lean core only (CLI + classical segmentation, no model)
	$(PIP) install -e .

install-web: | $(PY_BIN) ## Lean web UI (FastAPI; classical segmentation, no model)
	$(PIP) install -e ".[web]"

install-rembg: | $(PY_BIN) ## CPU rembg backend + web (purges onnxruntime-gpu first)
	-$(PIP) uninstall -y onnxruntime-gpu
	$(PIP) install -e ".[web,rembg]"

install-gpu: | $(PY_BIN) ## GPU rembg backend + web (onnxruntime-gpu + CUDA 12; purges CPU onnxruntime)
	-$(PIP) uninstall -y onnxruntime
	$(PIP) install -e ".[web,gpu]"
	@$(PY) -c "import onnxruntime as o; print('ORT providers:', o.get_available_providers())"

install-sam: | $(PY_BIN) ## SAM backend + web, CPU torch (override TORCH_INDEX for CUDA)
	$(PIP) install torch torchvision --index-url $(TORCH_INDEX)
	$(PIP) install -e ".[web,sam]"

install-sam-gpu: ## SAM backend + web on GPU (CUDA torch, cu126)
	$(MAKE) install-sam TORCH_INDEX=https://download.pytorch.org/whl/cu126

install-dev: $(ENV_DIR)/.stamp-dev frontend/node_modules/.stamp ## Full dev env (backend dev deps + frontend node deps)

download-sam: $(SAM_CKPT) ## Download the SAM vit_b checkpoint (~358 MB) if absent

# --------------------------------------------------------------------------- #
##@ Run (no dev tooling — just use the app)
# --------------------------------------------------------------------------- #
.PHONY: run run-gpu generate
run: $(ENV_DIR)/.stamp-web ## Install web+rembg (CPU) and launch the web UI  [HOST, PORT]
	$(PY) -m multiseedgen.web --host $(HOST) --port $(PORT)

run-gpu: install-gpu ## Install GPU rembg and launch the web UI  [HOST, PORT]
	$(PY) -m multiseedgen.web --host $(HOST) --port $(PORT)

generate: $(ENV_DIR)/.stamp-web ## Generate a dataset from CONFIG (default: soybean_healthy)
	$(PY) -m multiseedgen --config $(CONFIG)

# --------------------------------------------------------------------------- #
##@ Develop (backend + Vite dev server with hot reload)
# --------------------------------------------------------------------------- #
.PHONY: dev dev-backend dev-frontend
dev: $(ENV_DIR)/.stamp-dev frontend/node_modules/.stamp ## Run backend (:8000, --reload) + Vite (:5173) together
	@echo "Backend  -> http://$(HOST):$(PORT)   (FastAPI --reload)"
	@echo "Frontend -> http://localhost:5173    (Vite; proxies /api + /ws)"
	@bash -c 'set -m; \
	  $(PY) -m multiseedgen.web --host $(HOST) --port $(PORT) --reload & back=$$!; \
	  ( cd frontend && npm run dev ) & front=$$!; \
	  trap "kill -TERM $$back $$front 2>/dev/null; wait $$back $$front 2>/dev/null" INT TERM EXIT; \
	  wait -n $$back $$front'

dev-backend: $(ENV_DIR)/.stamp-dev ## Run only the FastAPI backend with --reload
	$(PY) -m multiseedgen.web --host $(HOST) --port $(PORT) --reload

dev-frontend: frontend/node_modules/.stamp ## Run only the Vite dev server
	cd frontend && npm run dev

# --------------------------------------------------------------------------- #
##@ Frontend (Node; the built bundle is committed, so this is dev-only)
# --------------------------------------------------------------------------- #
.PHONY: frontend-install frontend-build frontend-dev frontend-lint frontend-typecheck frontend-test frontend-check
frontend-install: frontend/node_modules/.stamp ## Install frontend node deps (npm ci)

frontend-build: frontend/node_modules/.stamp ## Rebuild the committed multiseedgen/web/static/ bundle
	cd frontend && npm run build

frontend-dev: frontend/node_modules/.stamp ## Vite dev server (alias of dev-frontend)
	cd frontend && npm run dev

frontend-lint: frontend/node_modules/.stamp ## ESLint the frontend
	cd frontend && npm run lint

frontend-typecheck: frontend/node_modules/.stamp ## Type-check the frontend (tsc --noEmit)
	cd frontend && npm run typecheck

frontend-test: frontend/node_modules/.stamp ## Run frontend unit tests (Vitest)
	cd frontend && npm run test

frontend-check: frontend/node_modules/.stamp ## Frontend CI gate: lint + typecheck + test + build
	cd frontend && npm run lint && npm run typecheck && npm run test && npm run build

# --------------------------------------------------------------------------- #
##@ Quality (mirrors .github/workflows/ci.yml)
# --------------------------------------------------------------------------- #
.PHONY: test test-all lint format format-check typecheck check ci
test: $(ENV_DIR)/.stamp-dev ## Run the portable test suite (skips gpu + golden)
	$(PY) -m pytest -m "not gpu and not golden"

test-all: $(ENV_DIR)/.stamp-dev ## Run the full test suite (includes slow/golden)
	$(PY) -m pytest

lint: $(ENV_DIR)/.stamp-dev ## Ruff lint
	$(PY) -m ruff check .

format: $(ENV_DIR)/.stamp-dev ## Ruff format + autofix (writes changes)
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

format-check: $(ENV_DIR)/.stamp-dev ## Ruff format check (no writes; CI mode)
	$(PY) -m ruff format --check .

typecheck: $(ENV_DIR)/.stamp-dev ## Mypy type-check
	$(PY) -m mypy multiseedgen

check: lint format-check typecheck test frontend-check ## Full local CI gate (backend + frontend)
ci: check ## Alias of check

# --------------------------------------------------------------------------- #
##@ Docker (GPU is optional; the default image is CPU and runs anywhere)
# --------------------------------------------------------------------------- #
.PHONY: docker-up docker-down docker-logs docker-gpu docker-sam docker-sam-gpu
docker-up: ## CPU web image, no GPU/toolkit needed -> http://localhost:8000
	docker compose -f $(COMPOSE_CPU) up --build

docker-down: ## Stop & remove the CPU stack
	docker compose -f $(COMPOSE_CPU) down

docker-logs: ## Follow the CPU stack logs
	docker compose -f $(COMPOSE_CPU) logs -f

docker-gpu: ## GPU rembg image (needs NVIDIA Container Toolkit) -> :8000
	docker compose -f $(COMPOSE_GPU) up --build

docker-sam: ## CPU SAM image -> :8001
	docker compose -f $(COMPOSE_GPU) --profile sam up --build multiseedgen-sam

docker-sam-gpu: ## GPU SAM image (needs toolkit) -> :8002
	docker compose -f $(COMPOSE_GPU) --profile sam-gpu up --build multiseedgen-sam-gpu

# --------------------------------------------------------------------------- #
##@ Clean (never touches user data: out/ datasets/ seeds/ models/)
# --------------------------------------------------------------------------- #
.PHONY: clean clean-venv clean-frontend clean-all
clean: ## Remove caches (pyc, pytest/mypy/ruff, coverage, egg-info)
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage .coverage.* htmlcov
	rm -rf multiseedgen.egg-info multiseedgen/*.egg-info

clean-venv: ## Remove the local ./.venv
	rm -rf $(VENV)

clean-frontend: ## Remove frontend node_modules + coverage
	rm -rf frontend/node_modules frontend/coverage

clean-all: clean clean-venv clean-frontend ## Remove caches, ./.venv, and frontend deps

# --------------------------------------------------------------------------- #
##@ Help
# --------------------------------------------------------------------------- #
.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mMultiSeedGen\033[0m — make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo
