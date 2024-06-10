SHELL = /bin/bash
VENV = venv-hnx
PYTHON3 = python3

############# Manage Environments #############

## Create environment using Pip
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

## Create environment using Poetry

.PHONY: install-with-poetry
install-with-poetry:
	poetry env remove --all
	poetry install --with test

## Create new requirements.txt from current poetry environment
.PHONY: requirements.txt
requirements.txt:
	poetry export --format requirements.txt --output requirements.txt --without-hashes --with test,widget,tutorials,docs

############# Tutorials #############
.PHONY: tutorials
tutorials:
	jupyter notebook tutorials

.PHONY: clean
clean:
	rm -rf .out .pytest_cache .tox *.egg-info dist build _build pytest.xml pytest_notebooks.xml .coverage

############# Running Tests, linters, formatters #############

## Tests
.PHONY: test
test:
	coverage run --source=hypernetx -m pytest tests
	coverage report -m

.PHONY: test-core
test-core:
	coverage run --source=hypernetx/classes -m pytest tests/classes --verbose
	coverage report -m


## Run test on Tox framework
## Includes linting, running tests on jupyter notebooks
.PHONY: test-tox
test-tox:
	@$(PYTHON3) -m tox --parallel

## Lint
.PHONY: lint
lint: pylint flake8

.PHONY: pylint
pylint:
	@$(PYTHON3) -m pylint --recursive=y --persistent=n --exit-zero --verbose hypernetx

# Todo: fix flake8 errors and remove --exit-zero
.PHONY: flake8
flake8:
	@$(PYTHON3) -m flake8 hypernetx --exit-zero

## precommit hooks that include formatter (Black)
.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files
