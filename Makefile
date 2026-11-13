.DEFAULT_GOAL := help

UV ?= uv
RUN = $(UV) run --no-sync

.PHONY: help setup format format-check lint check typecheck test smoke

help:
	@printf '%s\n' \
	  'make setup         Install the locked CPU and development dependencies' \
	  'make format        Sort imports and format Python files and notebook cells' \
	  'make format-check  Check imports and formatting without changing files' \
	  'make lint          Run all configured Ruff lint rules' \
	  'make check         Check formatting and lint' \
	  'make typecheck     Run mypy on both packages' \
	  'make test          Run the test suite for both tracks' \
	  'make smoke         Exercise both models and check the demo dependency'

setup:
	$(UV) sync --locked --extra cpu --group pretraining --group demo

format:
	$(RUN) ruff check --select I --fix .
	$(RUN) ruff format .

format-check:
	$(RUN) ruff check --select I .
	$(RUN) ruff format --check .

lint:
	$(RUN) ruff check .

check: format-check lint

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

smoke:
	$(RUN) python scripts/smoke_cpu.py --demo
