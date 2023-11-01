SHELL = /bin/bash

VENV = venv-hnx
PYTHON3 = python3


## Test

test: test-deps
	@$(PYTHON3) -m tox

test-ci: test-deps
	pre-commit install
	pre-commit run --all-files
	@$(PYTHON3) -m tox

test-ci-github: test-deps
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'
	@$(PYTHON3) -m tox

.PHONY: test, test-ci, test-ci-github

## Continuous Deployment
## Assumes that scripts are run on a container or test server VM

### Publish to PyPi
publish-deps:
	@$(PYTHON3) -m pip install -e .'[packaging]' --use-pep517

build-dist: publish-deps clean
	@$(PYTHON3) -m build --wheel --sdist
	@$(PYTHON3) -m twine check dist/*

## Assumes the following environment variables are set: TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL,
## See https://twine.readthedocs.io/en/stable/#environment-variables
publish-to-pypi: publish-deps build-dist
	@echo "Publishing to PyPi"
	$(PYTHON3) -m twine upload dist/*

.PHONY: build-dist publish-to-pypi publish-deps

### Update version

version-deps:
	@$(PYTHON3) -m pip install .'[releases]' --use-pep517

.PHONY: version-deps

### Documentation

docs-deps:
	@$(PYTHON3) -m pip install .'[documentation]' --use-pep517

.PHONY: docs-deps

## Tutorials

.PHONY: tutorial-deps
tutorial-deps:
	@$(PYTHON3) -m pip install .'[tutorials]' .'[widget]' --use-pep517

.PHONY: tutorials
tutorials:
	jupyter notebook tutorials



## Environment

clean-venv:
	rm -rf $(VENV)

clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build

venv: clean-venv
	@$(PYTHON3) -m venv $(VENV);

test-deps:
	@$(PYTHON3) -m pip install .'[testing]' --use-pep517

all-deps:
	@$(PYTHON3) -m pip install -e .'[all]' --use-pep517

.PHONY: clean clean-venv venv all-deps test-deps
