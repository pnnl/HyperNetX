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
## Includes linting, running tests on jupyter notebooks, building and checking the documentation
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

.PHONY: test-ci-stash
test-ci-stash: install-poetry
	poetry run tox --parallel

.PHONY: docs-deps
docs-deps: install-poetry
	poetry run tox -e build-docs

# TODO: fix/update Github Actions
.PHONY: test-ci-github
test-ci-github: ci-github-deps test-tox

.PHONY: github-ci-deps
ci-github-deps:
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'



## Continuous Deployment
## Assumes that scripts are run on a container or test server VM
### Publish to PyPi

.PHONY: build-dist
build-dist: clean
	@$(PYTHON3) -m pip install build twine
	@$(PYTHON3) -m build --wheel --sdist
	@$(PYTHON3) -m twine check dist/*

## Assumes the following environment variables are set: TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL,
## See https://twine.readthedocs.io/en/stable/#environment-variables
.PHONY: publish-to-pypi
publish-to-pypi: build-dist
	@echo "Publishing to PyPi"
	$(PYTHON3) -m twine upload dist/*

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
	rm -rf .out .pytest_cache .tox *.egg-info dist build
