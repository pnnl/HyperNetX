********************
Installing HyperNetX
********************

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

    >>> conda create -n venv-hnx python=3.8 -y
    >>> conda activate venv-hnx

Using venv
*************************

    >>> python -m venv venv-hnx
    >>> source venv-hnx/bin/activate


Using virtualenv
*************************

    >>> virtualenv venv-hnx
    >>> source venv-hnx/bin/activate


For Windows Users
******************

On both Windows PowerShell or Command Prompt, you can use the following command to activate your virtual environment:

    >>> .\env-hnx\Scripts\activate


To deactivate your environment, use:

    >>> .\env-hnx\Scripts\deactivate


Installation
############

Regardless of how you install HyperNetX, ensure that your environment is activated and that you are running Python >=3.8.

Installing from PyPi
********************

    >>> pip install hypernetx

If you want to use supported applications built upon HyperNetX (e.g. ``hypernetx.algorithms.hypergraph_modularity`` or
``hypernetx.algorithms.contagion``), you can install HyperNetX with those supported applications by using
the following command:

    >>> pip install igraph celluloid

Installing from Source
**********************

Ensure that you have ``git`` installed.

    >>> git clone https://github.com/pnnl/HyperNetX.git
    >>> cd HyperNetX
    >>> make venv
    >>> source venv-hnx/bin/activate
    >>> make install


Post-Installation Actions
#########################

Interact with HyperNetX in a REPL
*********************************

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
**************************************

If you have installed HyperNetX from source, you can perform additional actions such as viewing the provided Jupyter notebooks
or building the documentation locally.

Ensure that you have activated your virtual environment and are at the root of the source directory before running any of the following commands:


Viewing jupyter notebooks
--------------------------

The following command will automatically open the notebooks in a browser.

    >>> make tutorials


Building documentation
-----------------------

The following commands will build and open a local version of the documentation in a browser:

    >>> pip install sphinx sphinx-autobuild sphinx-rtd-theme sphinx-copybutton
    >>> cd docs
    >>> make html
    >>> open build/index.html
