SHELL = /bin/bash

VENV = venv-hnx
PYTHON_VENV = $(VENV)/bin/python3
PYTHON3 = python3


## Test

test: test-deps
	@$(PYTHON3) -m tox

test-ci: test-deps
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'
	pre-commit install
	pre-commit run --all-files
	@$(PYTHON3) -m tox -e py38 -r
	@$(PYTHON3) -m tox -e py38-notebooks -r

test-ci-github: test-deps
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'
	@$(PYTHON3) -m tox

test-coverage: test-deps
	coverage run --source=hypernetx -m pytest
	coverage html

.PHONY: test, test-ci, test-ci-github, test-coverage

## Continuous Deployment
## Assumes that scripts are run on a container or test server VM

### Publish to PyPi
publish-deps:
	@$(PYTHON3) -m pip install -e .'[packaging]'

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
	@$(PYTHON3) -m pip install .'[releases]'

.PHONY: version-deps

#### Documentation

docs-deps:
	@$(PYTHON3) -m pip install -e .'[documentation]' --use-pep517

.PHONY: docs-deps

## Environment

clean-venv:
	rm -rf $(VENV)

clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build

venv: clean-venv
	@$(PYTHON3) -m venv $(VENV);

test-deps:
	@$(PYTHON3) -m pip install -e .'[testing]' --use-pep517

all-deps:
	@$(PYTHON3) -m pip install -e .'[all]' --use-pep517

.PHONY: clean clean-venv venv all-deps test-deps
