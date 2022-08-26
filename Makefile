SHELL = /bin/bash

VENV = .venv_test
PYTHON = $(VENV)/bin/python3
PYTHON3 = python3

## Environment

venv:
	@python3 -m venv $(VENV);

.PHONY: venv

## Test

test: clean venv
	@$(PYTHON) -m pip install -e .'[auto-testing]'
	@$(PYTHON) -m tox

test-ci:
	@$(PYTHON3) -m pip install -e .'[auto-testing]'
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'
	pre-commit install
	pre-commit run --all-files
	@$(PYTHON3) -m tox

.PHONY: test, test-ci

## Build package

build-dist: clean
	@$(PYTHON3) -m pip install -e .'[packaging]'
	@$(PYTHON3) -m build --wheel --sdist
	@$(PYTHON3) -m twine check dist/*


publish-to-test-pypi:
	@echo "Publishing to testpypi"
	# $(PYTHON) -m twine upload --repository testpypi dist/*

publish-to-pypi:
	@echo "Publishing to PyPi"
	# $(PYTHON) -m twine upload --repository pypi dist/*

.PHONY: build-dist publish-to-test-pypi publish-to-pypi

clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build $(VENV)

.PHONY: clean
