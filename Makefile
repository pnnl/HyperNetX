SHELL = /bin/bash

VENV = venv-hnx
PYTHON3 = python3


## Lint

.PHONY: lint
lint: pylint flake8 mypy

.PHONY: pylint
pylint:
	@$(PYTHON3) -m pylint --recursive=y --persistent=n --verbose hypernetx

.PHONY: mypy
mypy:
	@$(PYTHON3) -m mypy hypernetx || true

.PHONY: flake8
flake8:
	@$(PYTHON3) -m flake8 hypernetx --exit-zero

.PHONY: format
format:
	@$(PYTHON3) -m black hypernetx


## Tests

.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

.PHONY: test
test:
	coverage run --source=hypernetx -m pytest
	coverage report -m

.PHONY: test-ci
test-ci:
	@$(PYTHON3) -m tox

.PHONY: test-ci-stash
test-ci-stash: lint-deps lint pre-commit test-deps test-ci


.PHONY: test-ci-github
test-ci-github: lint-deps lint pre-commit ci-github-deps test-deps test-ci


## Continuous Deployment
## Assumes that scripts are run on a container or test server VM
### Publish to PyPi

.PHONY: publish-deps
publish-deps:
	@$(PYTHON3) -m pip install -e .[packaging] --use-pep517

.PHONY: build-dist
build-dist: publish-deps clean
	@$(PYTHON3) -m build --wheel --sdist
	@$(PYTHON3) -m twine check dist/*

## Assumes the following environment variables are set: TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL,
## See https://twine.readthedocs.io/en/stable/#environment-variables
.PHONY: publish-to-pypi
publish-to-pypi: publish-deps build-dist
	@echo "Publishing to PyPi"
	$(PYTHON3) -m twine upload dist/*


### Update version

.PHONY: version-deps
version-deps:
	@$(PYTHON3) -m pip install .[releases] --use-pep517


### Documentation

.PHONY: docs-deps
docs-deps:
	@$(PYTHON3) -m pip install .[documentation] --use-pep517


## Tutorials

.PHONY: tutorial-deps
tutorial-deps:
	@$(PYTHON3) -m pip install .[tutorials] .[widget] --use-pep517

.PHONY: tutorials
tutorials:
	jupyter notebook tutorials


## Environment

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: clean
clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build

.PHONY: venv
venv: clean-venv
	@$(PYTHON3) -m venv $(VENV);

.PHONY: github-ci-deps
ci-github-deps:
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'

.PHONY: lint-deps
lint-deps:
	@$(PYTHON3) -m pip install .[lint] --use-pep517

.PHONY: format-deps
format-deps:
	@$(PYTHON3) -m pip install .[format] --use-pep517

.PHONY: test-deps
test-deps:
	@$(PYTHON3) -m pip install .[testing] --use-pep517

.PHONY: all-deps
all-deps:
	@$(PYTHON3) -m pip install .[all] --use-pep517
