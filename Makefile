SHELL = /bin/bash
VENV = venv-hnx
PYTHON3 = python3

## Lint
.PHONY: lint
lint: pylint flake8

.PHONY: pylint
pylint:
	@$(PYTHON3) -m pylint --recursive=y --persistent=n --verbose hypernetx

# Todo: fix flake8 errors and remove --exit-zero
.PHONY: flake8
flake8:
	@$(PYTHON3) -m flake8 hypernetx --exit-zero

.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

## Tests
.PHONY: test
test:
	coverage run --source=hypernetx -m pytest
	coverage report -m

## Tests using Tox
## Includes linting, running tests on jupyter notebooks
.PHONY: test-tox
test-tox:
	@$(PYTHON3) -m tox --parallel

### Tests using Poetry + Tox
### Used by Bamboo CI Pipeline, Github Workflows CI Pipeline
.PHONY: install-poetry
install-poetry:
	pip install poetry==1.8.2
	poetry config virtualenvs.in-project true
	poetry run pip install tox

.PHONY: run-poetry-tox
run-poetry-tox:
	poetry run tox --parallel

.PHONY: clean-poetry
clean-poetry:
	poetry env remove --all

.PHONY: test-ci-stash
test-ci-stash: install-poetry run-poetry-tox clean-poetry

.PHONY: test-ci-github
test-ci-github: run-poetry-tox

.PHONY: run-poetry-tox-build-docs
run-poetry-tox-build-docs:
	poetry run tox -e build-docs

.PHONY: build-docs
build-docs: install-poetry run-poetry-tox-build-docs clean-poetry



## Publish to PyPi
## Targets are included as a backup in case the Github Workflows CI can't publish to PyPi and we need to do it manually
## Assumes the following environment variables are set: PYPI_API_TOKEN
.PHONY: publish-to-pypi
publish-to-pypi: build-dist
	@echo "Publishing to PyPi"
	poetry config pypi-token.pypi PYPI_API_TOKEN
	poetry config repositories.pypi https://pypi.org/simple/
	poetry publish --dry-run
	#poetry publish

.PHONY: build-dist
build-dist: clean
	poetry run pip install twine
	poetry build
	poetry run twine check dist/*


## Tutorials
.PHONY: tutorials
tutorials:
	jupyter notebook tutorials


## Environment
.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: venv
venv: clean-venv
	@$(PYTHON3) -m venv $(VENV);

.PHONY: install-reqs
install-reqs:
	@$(PYTHON3) -m pip install -r requirements.txt


## Clean
.PHONY: clean
clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build _build
