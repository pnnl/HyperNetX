HyperNetX
==========

<img src="https://raw.githubusercontent.com/pnnl/HyperNetX/master/docs/source/images/harrypotter_basic_hyp.png" align="right" width="300pt">

[![Pytest](https://github.com/pnnl/HyperNetX/actions/workflows/ci.yml/badge.svg)](https://github.com/pnnl/HyperNetX/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![CITATION.cff](https://github.com/pnnl/HyperNetX/actions/workflows/cff.yml/badge.svg)](https://github.com/pnnl/HyperNetX/actions/workflows/cff.yml)

The HyperNetX library provides classes and methods for the analysis
and visualization of complex network data modeled as hypergraphs.
The library generalizes traditional graph metrics.

HypernetX was developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principal Developer and Designer: Brenda Praggastis
* Development Team: Audun Myers, Mark Bonicillo
* Visualization: Dustin Arendt, Ji Young Yun
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Principal Contributors (Design, Theory, Code): Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Helen Jenne, Cliff Joslyn, Nicholas Landry, Audun Myers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Greg Roek, Mirah Shi, Francois Theberge, Ji Young Yun

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: https://pnnl.github.io/HyperNetX

For questions and comments contact the developers directly at: hypernetx@pnnl.gov

Summary - Release highlights - HNX 2.3
--------------------------------------

HyperNetX 2.3. is the latest, stable release. The core library has been refactored to take better advantage
of Pandas Dataframes, improve readability and maintainability, address bugs, and make it easier to change.
New features have been added, most notably the ability to add and remove edges, nodes, and incidences.

**Version 2.3 is not backwards compatible. Objects constructed using earlier versions
can be imported using their incidence dictionaries and/or property datafames.**

What's New
----------
1. Hypergraph now supports adding and removing edges, nodes, and incidences
1. Hypergraph also supports the sum, difference, union, and intersection of a Hypergraph to another Hypergraph
1. New factory methods to support the Hypergraph constructor
1. EntitySet has been replaced by HypergraphView
1. IncidenceStore and PropertyStore are new classes that maintain the structure and attributes of a Hypergraph
1. Hypergraph constructors accept cell, edge, and node metadata.


What's Changed
--------------
1. HNX now requires Python ">=3.10,<4.0.0"
1. HNX core libraries have been updated
1. Updated tutorials
1. The `static` and `dynamic` distinctions no longer exist. All hypergraphs use the same underlying data structure, supported by Pandas dataFrames. All hypergraphs maintain a `state_dict` to avoid repeating computations.
1. The `nwhy` optimizations are no longer supported.


Tutorials Available for Colab
=============================

Google Colab
------------


<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%201%20-%20HNX%20Basics.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 1 - HNX Basics</span>
</a>
<br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%202%20-%20Visualization%20Methods.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 2 - Visualization Methods</span>
</a>
<br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%203%20-%20LesMis%20Case%20Study.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 3 - LesMis Case Study</span>
</a>
<br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%204%20-%20LesMis%20Visualizations-BookTour.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 4 - LesMis Visualizations-Book Tour</span>
</a>
<br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%205%20-%20HNX%20attributed%20hypergraph.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 5 - HNX attributed hypergraph</span>
</a>
<br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/basic/Basic%206%20-%20Hypergraph%20Arithmetic.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Basic 6 - Hypergraph Arithmetic.ipynb</span>
</a>
<br>


Jupyter Notebooks
-----------------

Additional tutorials that can be run as Jupyter Notebooks are found under [tutorials](./tutorials).

Installation
====================

The recommended installation method for most users is to create a virtual environment and install HyperNetX from PyPi.

HyperNetX may be cloned or forked from [GitHub](https://github.com/pnnl/HyperNetX).

Prerequisites
-------------
HyperNetX officially supports Python >=3.10,<4.0.0

Create a virtual environment
----------------------------

### Using venv


```shell
python -m venv venv-hnx
source venv-hnx/bin/activate
```


### Using Anaconda


```shell
conda create -n venv-hnx python=3.11 -y
conda activate venv-hnx
```


### Using virtualenv


```shell
virtualenv venv-hnx
source venv-hnx/bin/activate
```


### For Windows Users

On both Windows PowerShell or Command Prompt, you can use the following command to activate your virtual environment:

```shell
.\env-hnx\Scripts\activate
```

To deactivate your environment, use:

```shell
.\env-hnx\Scripts\deactivate
```

Installing HyperNetX
====================

Regardless of how you install HyperNetX, ensure that your environment is activated and that you are running Python ">=3.10,<4.0.0".


Installing from PyPi
--------------------

```shell
pip install hypernetx
```

Installing from Source
----------------------

Ensure that you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed.

```shell
git clone https://github.com/pnnl/HyperNetX.git
cd HyperNetX

# Create a virtual environment
make venv
source venv-hnx/bin/activate

# install required dependencies
make install
```

# Using HyperNetX on Docker

As an alternative to installing HyperNetX, you can use the officially supported HyperNetX Docker image maintained at
[DockerHub](https://hub.docker.com/r/hypernetx/hypernetx). Use the image to quickly start HyperNetX in a Docker container.
The container starts a Jupyter Notebook that has the latest version of HyperNetX and HNXWidget installed;
it also contains all the HyperNetX tutorials.

## Run the Container

* Using Docker CLI, run the container in the foreground:

```
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work hypernetx/hypernetx:latest
```

* Alternatively, you can create a `docker-compose.yml` file with the following:
```
version: '3'

services:
  hypernetx:
    image: hypernetx/hypernetx:latest
    ports:
      - "8888:8888"
    tty: true
    stdin_open: true
    volumes:
      - "${PWD}:/home/jovyan/work"
```

Once `docker-compose.yml` is created, run the container:

```
docker-compose up
```

## Open Jupyter Notebook

After the container has started, access the HyperNetX Jupyter Notebooks by opening the following URL in a browser:

[http://localhost:8888/tree](http://localhost:8888/tree)


# Development

As a developer, set up your environment using either the standard `pip` tool or [`Poetry`](https://python-poetry.org/).

## Using Pip

### Setup virtual environment and install HNX

Create a virtual environement. Then install an editable version of HNX and also install additional dependencies to support testing and jupyter notebooks:
```
python -m venv venv-hnx
source venv-hnx/bin/activate
pip install -e .
pip install -r requirements.txt
```

As an alternative, you can also install all these requirements in one Make target:

```
make venv
source venv-hnx/bin/activate
make install
```

### Setup pre-commit

Use the [pre-commit framework](https://pre-commit.com/) to automatically point out issues and resolve those issues before code review.
It is highly recommended to install pre-commit in your development environment so that issues with your code can be found before you submit a
pull request. More importantly, using pre-commit will automatically format your code changes so that they pass the CI build. For example, pre-commit will
automatically run the formatter Black on your code changes.

```shell
# Once installed, pre-commit will be triggered every time you make a commit in your environment
pre-commit install
```


## Using Poetry

This library uses [Poetry](https://python-poetry.org/docs/) to manage dependencies and packaging. Poetry can also be
used to manage your environment for development.

### Prerequisites

* [Install Poetry](https://python-poetry.org/docs/#installation)


### Configure Poetry

[Configure your Poetry](https://python-poetry.org/docs/configuration/) to ensure that the virtual environment gets created in your project directory (this is not necessary but recommended for convenience):

```
poetry config virtualenvs.in-project true

# check the poetry configuration
poetry config --list
```

### Setup virtual environment and install HNX

Create and activate a virtual environment.

```
poetry shell
```

Install HyperNetX in editable mode, the library's core/required dependencies, and the optional dependencies to support development.

```
poetry install --with test,lint,docs,release,tutorials
```

Details about these dependencies are defined in [pyproject.toml](pyproject.toml).

### Setup Pre-commit

Use the [pre-commit framework](https://pre-commit.com/) to automatically point out issues and resolve those issues before code review.
It is highly recommended to install pre-commit in your development environment so that issues with your code can be found before you submit a
pull request. More importantly, using pre-commit will automatically format your code changes so that they pass the CI build. For example, pre-commit will
automatically run the formatter Black on your code changes.

```shell
# Once installed, pre-commit will be triggered every time you make a commit in your environment
pre-commit install
```

### Details about optional dependencies

#### Install support for testing


> ℹ️ **NOTE:** This project has pytest configuration contained in pyproject.toml. By default, pytest will use those configuration settings to run tests.

```shell
poetry install --with test

# activate your virtual environment created by poetry
poetry shell

# run tests
python -m pytest

# run tests and show coverage report
python -m pytest --cov=hypernetx

# Generate an HTML code coverage report and view it on a browser
coverage html
open htmlcov/index.html
```

#### Install support for tutorials

```shell
poetry install --with tutorials

# activate your virtual environment created by poetry
poetry shell

# open Jupyter notebooks in a browser
make tutorials
```

#### Code Quality: Pylint, Black

HyperNetX uses a number of tools to maintain code quality:

* Pylint
* Black

Before using these tools, ensure that you install Pylint in your environment:

```shell
poetry install --with lint

# activate your virtual environment created by poetry
poetry shell
```


[Pylint](https://pylint.pycqa.org/en/latest/index.html) is a static code analyzer for Python-based projects. From the [Pylint docs](https://pylint.pycqa.org/en/latest/index.html#what-is-pylint):

> Pylint analyses your code without actually running it. It checks for errors, enforces a coding standard, looks for code smells, and can make suggestions about how the code could be refactored. Pylint can infer actual values from your code using its internal code representation (astroid). If your code is import logging as argparse, Pylint will know that argparse.error(...) is in fact a logging call and not an argparse call.

To run Pylint and view the results of Pylint, run the following command:

```shell
pylint hypernetx
```

You can also run Pylint on the command line to generate a report on the quality of the codebase and save it to a file named "pylint-results.txt":

```shell
pylint hypernetx --output=pylint-results.txt
```

For more information on configuration, see https://pylint.pycqa.org/en/latest/user_guide/configuration/index.html

[Black](https://black.readthedocs.io/en/stable/) is a PEP 8 compliant formatter for Python-based project. This tool is highly opinionated about how Python should be formatted and will automagically reformat your code.


```shell
black hypernetx
```

### Documentation

Build and view documentation locally:

```shell
poetry install --with docs

# activate your virtual environment created by poetry
poetry shell

cd docs
make html
open docs/build/html/index.html
```

When editing documentation, you can auto-rebuild the documentation locally so that you can view your document changes
live on the browser without having to rebuild every time you have a change.

```shell
cd docs
make livehtml
```

This make script will run in the foreground on your terminal. You should see the following:

```shell
The HTML pages are in docs/html.
[I 230324 09:50:48 server:335] Serving on http://127.0.0.1:8000
[I 230324 09:50:48 handlers:62] Start watching changes
[I 230324 09:50:48 handlers:64] Start detecting changes
[I 230324 09:50:54 handlers:135] Browser Connected: http://127.0.0.1:8000/install.html
[I 230324 09:51:02 handlers:135] Browser Connected: http://127.0.0.1:8000/
```

Click on [http://127.0.0.1:8000/install.html](http://127.0.0.1:8000/install.html) to open the docs on your browser. Since this will auto-rebuild, every time
you change a document file, it will automatically render on your browser, allowing you to verify your document changes.


## Developing and Testing the Docker Image

If you want to test the Docker image after making any source code changes, follow this workflow:

1. Make a change in the HNX codebase
2. Build image for multi-platforms (i.e.ARM64, x86): `docker build --platform linux/amd64,linux/arm64 --rm --tag hypernetx/hypernetx:latest .`
   3. If you're having issues building, see https://docs.docker.com/desktop/containerd/
3. Test image: `docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work hypernetx/hypernetx:latest`
4. Open a browser to [http://localhost:8888/tree](http://localhost:8888/tree). Check that tutorials still work and/or open a notebook and test the changes that you made.
5. Once finished testing, kill the container using Ctrl-C


Notice
======
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<div>
  <pre style="text-align: center; font-size: 10pt;">
    PACIFIC NORTHWEST NATIONAL LABORATORY
    operated by
    BATTELLE
    for the
    UNITED STATES DEPARTMENT OF ENERGY
    under Contract DE-AC05-76RL01830
  </pre>
</div>



License
=======

Released under the [3-Clause BSD license](LICENSE.rst)
