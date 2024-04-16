SHELL = /bin/bash
VENV = venv-hnx
PYTHON3 = python3

############# Manage Environments #############

## Environment using Pip
.PHONY: venv
venv: clean-venv
	@$(PYTHON3) -m venv $(VENV);

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: install
install:
	@$(PYTHON3) -m pip install --upgrade pip
	@$(PYTHON3) -m pip install -e .
	@$(PYTHON3) -m pip install -r requirements.txt

## Environment using Poetry

.PHONY: develop
develop: clean-poetry-env
	poetry shell
	poetry install --with test


.PHONY: requirements.txt
requirements.txt:
	poetry export --format requirements.txt --output requirements.txt --without-hashes --with test,widget,tutorials,docs

############# Running Tests, linters #############

## Tests
.PHONY: test
test:
	coverage run --source=hypernetx -m pytest tests
	coverage report -m

.PHONY: test-core
test-core:
	coverage run --source=hypernetx/classes -m pytest tests/classes --verbose
	coverage report -m


## Tests using Tox
## Includes linting, running tests on jupyter notebooks
.PHONY: test-tox
test-tox:
	@$(PYTHON3) -m tox --parallel

### Tests using Poetry + Tox
### Used by Bamboo CI Pipeline, Github Workflows CI Pipeline

.PHONY: test-ci-stash
test-ci-stash: install-poetry-stash run-poetry-tox clean-poetry-env

.PHONY: build-docs-stash
build-docs-stash: install-poetry-stash run-build-docs clean-poetry-env

.PHONY: install-poetry-stash
install-poetry-stash:
	pip install poetry==1.8.2 tox

.PHONY: run-poetry-tox
run-poetry-tox:
	tox --parallel

.PHONY: run-build-docs
run-build-docs:
	tox -e build-docs

.PHONY: clean-poetry-env
clean-poetry-env:
	poetry env remove --all

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

############# Packaging and Publishing to PyPi #############
## Uses Poetry to manage packaging and publishing
## Targets are included as a backup in case the Github Workflows CI can't publish to PyPi and we need to do it manually
## Assumes the following environment variables are set: PYPI_API_TOKEN
.PHONY: publish-to-pypi
publish-to-pypi: build check-long-desc
	@echo "Publishing to PyPi"
	poetry config pypi-token.pypi PYPI_API_TOKEN
	poetry config repositories.pypi https://pypi.org/simple/
	poetry publish --dry-run
	#poetry publish

.PHONY: build
build: clean
	poetry build

.PHONY: check-long-desc
check-long-desc:
	poetry run pip install twine
	poetry run twine check dist/*

############# Misc #############
.PHONY: tutorials
tutorials:
	jupyter notebook tutorials

.PHONY: clean
clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build _build pytest.xml pytest_notebooks.xml .coverage
