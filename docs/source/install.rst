********************
Installing HyperNetX
********************


Installation
############

The recommended installation method for most users is to create a virtual environment
and install HyperNetX from PyPi.

.. _Github:  https://github.com/pnnl/HyperNetX

HyperNetX may be cloned or forked from Github_.


Prerequisites
######################

HyperNetX officially supports Python 3.8, 3.9, 3.10 and 3.11.


Create a virtual environment
############################

Using Anaconda
*************************

    >>> conda create -n env-hnx python=3.8 -y
    >>> conda activate env-hnx

Using venv
*************************

    >>> python -m venv venv-hnx
    >>> source env-hnx/bin/activate


Using virtualenv
*************************

    >>> virtualenv env-hnx
    >>> source env-hnx/bin/activate


For Windows Users
******************

On both Windows PowerShell or Command Prompt, you can use the following command to activate your virtual environment:

    >>> .\env-hnx\Scripts\activate


To deactivate your environment, use:

    >>> .\env-hnx\Scripts\deactivate


Installing Hypernetx
####################

Regardless of how you install HyperNetX, ensure that your environment is activated and that you are running Python >=3.8.

Installing from PyPi
*************************

    >>> pip install hypernetx


Installing from Source
*************************

Ensure that you have ``git`` installed.

    >>> git clone https://github.com/pnnl/HyperNetX.git
    >>> cd HyperNetX
    >>> pip install -e .['all']

If you are using zsh as your shell, ensure that the single quotation marks are placed outside the square brackets:

    >>> pip install -e .'[all]'


Post-Installation Actions
##########################

Running Tests
**************

    >>> python -m pytest

Interact with HyperNetX in a REPL
********************************************

Ensure that your environment is activated and that you run ``python`` on your terminal to open a REPL:

    >>> import hypernetx as hnx
    >>> data = { 0: ('A', 'B'), 1: ('B', 'C'), 2: ('D', 'A', 'E'), 3: ('F', 'G', 'H', 'D') }
    >>> H = hnx.Hypergraph(data)
    >>> list(H.nodes)
    ['G', 'F', 'D', 'A', 'B', 'H', 'C', 'E']
    >>> list(H.edges)
    [0, 1, 2, 3]
    >>> H.shape
    (8, 4)


Other Actions if installed from source
********************************************

Ensure that you are at the root of the source directory before running any of the following commands:

Viewing jupyter notebooks
--------------------------

The following command will automatically open the notebooks in a browser.

    >>> jupyter-notebook tutorials


Building documentation
-----------------------

The following commands will build and open a local version of the documentation in a browser:

    >>> make build-docs
    >>> open docs/build/index.html


