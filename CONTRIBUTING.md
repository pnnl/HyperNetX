# Contributing to HyperNetX

## Code of Conduct

We want this community to be friendly and respectful to each other. Please read [the full text](CODE_OF_CONDUCT.md) so that you can understand what actions will and will not be tolerated.

## Requirements

- Python >=3.8,<3.12

## Our Development Process

### Development workflow

> **Working on your first pull request?** You can learn how from this *free* series: [How to Contribute to an Open Source Project on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project).

1. Fork the repo and create your branch from the `develop` branch. Here's a guide on [how to fork a repository](https://help.github.com/articles/fork-a-repo/).
1. Create a Python virtual environment. For convenience, our Makefile provides a target called 'venv' that will create a virtual environment for you. Run the following command: `make venv`
1. Activate the virtual environment. If you used the Makefile target in the previous step, activate the virtual environment by the running the following command: `source venv-hnx/bin/activate`
1. Install the library in development mode: `pip install -e .`
1. Install testing dependencies: `pip install -e .['testing'] `
1. Do the changes you want and ensure all tests pass by running `python -m pytest` before sending a pull request.

### Commit message convention

Ensure that commit messages begin with a verb and are in the present tense. Write meaningful commit messages that concisely describes the changes in a commit.
Read this blog post, [How to Write a Git Commit Message](https://cbea.ms/git-commit/) for some guidance.

### Linting and tests

 We use `pylint` and `black` for linting and formatting the code, and `pytest` for testing.

We have included a pre-commit config file that can be used to install pre-commit hooks that will lint your code changes every time you create a commit.
To install pre-commit, you must first install the testing dependencies: `pip install -e .['testing']`. Then run: `pre-commit install`.

Once installed, every time you create a new commit, the linters and formatters will run on the changed code. It is highly recommended to use these `pre-commit` hooks
because these same hooks are run in our CI/CD pipelines on every pull request. Catching pre-commit issues early will prevent CI/CD pipeline issues on your pull request.

### Sending a pull request

- Prefer small pull requests focused on one change.
- Verify that and all tests are passing.
- Verify all in-code documentation is correct (it will be used to generate API documentation).


## Report an issue, bug, or feature request

Here are the [steps to creating an issue on GitHub](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart).  When reporting a bug,

- search for related issues on GitHub. You might be able to get answer without the hassle of creating an issue
- describe the current behavior and explain which behavior you expected to see instead and why. At this point you can also tell which alternatives do not work for you.
  - (if applicable) provide error messages
  - (if applicable) provide a step by step description of the problem; if possible include code that others can use to reproduce it
  - You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
  - provide a clear, specific title
  - include details on your setup (operating system, python version, etc.)
- use the most recent version of this library and the source language (e.g. Python); that fixes a lot of problems
- here are [more details on getting the most out of issue reporting!](https://marker.io/blog/how-to-write-bug-report)


## Where can I go for help?

If you're stuck or don't know where to begin, then you're in good company -- we've all been there!  We're here to help, and we'd love to hear from you:

- open an issue report on [GitHub](https://github.com/pnnl/HyperNetX/issues)
- email us at [hypernetx@pnnl.gov](mailto:hypernetx@pnnl.gov)
