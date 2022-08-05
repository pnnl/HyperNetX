
SHELL = /bin/bash

# Variables
VENV = venv_test
PYTHON = $(VENV)/bin/python

## Environment

venv:
	@rm -rf VENV;
	@python -m venv $(VENV);

deps:
	@$(PYTHON) -m pip install tox

.PHONY: venv deps

## Test

test: venv deps
	rm -rf .tox
	@$(PYTHON) -m tox

.PHONY: test

clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build

.PHONY: clean