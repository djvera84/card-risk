PYTHON ?= .venv/bin/python
BIN := $(dir $(PYTHON))

.PHONY: install test lint format data features fraud tune underwriting simulator notebooks all

install:
	python3 -m venv .venv
	$(BIN)pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(BIN)ruff check src tests
	$(BIN)ruff format --check src tests

format:
	$(BIN)ruff check --fix src tests
	$(BIN)ruff format src tests

# ---- Fraud model on the transactions data ----
data:
	$(BIN)fraud-prepare-data

features: data
	$(BIN)fraud-build-features

fraud: features
	$(BIN)fraud-baselines
	$(BIN)fraud-train
	$(BIN)fraud-evaluate

# Optional and slow: writes artifacts/best_params.json, which fraud-train then uses.
tune: features
	$(BIN)fraud-tune

# ---- Underwriting model and credit policy simulator (simulated applicants) ----
underwriting:
	$(BIN)underwriting-simulate
	$(BIN)underwriting-train

simulator: underwriting
	$(BIN)credit-simulator

notebooks:
	$(BIN)jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

all: fraud simulator notebooks
