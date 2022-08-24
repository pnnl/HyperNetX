SHELL = /bin/bash

VENV = .venv_test
PYTHON = $(VENV)/bin/python

## Environment

venv:
	@python3 -m venv $(VENV);

deps-testing:
	@$(PYTHON) -m pip install -e .'[testing]'

deps-packaging:
	@$(PYTHON) -m pip install -e .'[packaging]'

.PHONY: venv deps-testing deps-packaging

## Test

test: clean venv deps-testing
	@$(PYTHON) -m tox

.PHONY: test

## Build package

build: clean venv deps-packaging
	@$(PYTHON) -m build --wheel --sdist
	@$(PYTHON) -m twine check dist/*


publish-to-test-pypi:
	@echo "Publishing to testpypi"
	# $(PYTHON) -m twine upload --repository testpypi dist/*

publish-to-pypi:
	@echo "Publishing to PyPi"
	# $(PYTHON) -m twine upload --repository pypi dist/*

.PHONY: build publish-to-test-pypi

clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build $(VENV)

.PHONY: clean