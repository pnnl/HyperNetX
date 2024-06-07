********************
Installing HyperNetX
********************

The recommended installation method for most users is to create a virtual environment
and install HyperNetX from PyPi.

.. _Github:  https://github.com/pnnl/HyperNetX

HyperNetX may be cloned or forked from Github_.


Prerequisites
#############

HyperNetX officially supports Python >=3.10,<4.0.0.


Create a virtual environment
############################

Using Anaconda
*************************

.. code-block:: bash

    conda create -n venv-hnx python=3.11 -y
    conda activate venv-hnx

Using venv
*************************

.. code-block:: bash

    python -m venv venv-hnx
    source venv-hnx/bin/activate


Using virtualenv
*************************

.. code-block:: bash

    virtualenv venv-hnx
    source venv-hnx/bin/activate


For Windows Users
******************

On both Windows PowerShell or Command Prompt, you can use the following command to activate your virtual environment:

.. code-block::

    .\env-hnx\Scripts\activate


To deactivate your environment, use:

.. code-block::

    .\env-hnx\Scripts\deactivate


Installation
############

After activating your virtual environment, install HyperNetX.

Installing from PyPi
********************

.. code-block:: bash

    pip install hypernetx


Installing from Source
**********************

The source code provides a Makefile to simplify the installation process. Ensure that you have ``make`` and ``git`` installed.

.. code-block:: bash

    git clone https://github.com/pnnl/HyperNetX.git
    cd HyperNetX
    make venv
    source venv-hnx/bin/activate
    make install



Post-Installation Actions
#########################

Interact with HyperNetX in a REPL
*********************************

Ensure that your environment is activated and that you run ``python`` on your terminal to open a REPL:

    >>> import hypernetx as hnx
    >>> data = { 0: ('A', 'B'), 1: ('B', 'C'), 2: ('D', 'A', 'E'), 3: ('F', 'G', 'H', 'D') }
    >>> H = hnx.Hypergraph(data)
    >>> list(H.nodes)
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
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

.. code-block:: bash

    make tutorials


Building documentation
-----------------------

The following commands will build and open a local version of the documentation in a browser:

.. code-block:: bash

    cd docs
    make html
    open build/index.html



Using HyperNetX on Docker
#########################

.. _DockerHub: https://hub.docker.com/r/hypernetx/hypernetx

As an alternative to installing HyperNetX, you can use the officially supported HyperNetX Docker image maintained at DockerHub_.
Use the image to quickly start HyperNetX in a Docker container. The container starts a Jupyter Notebook that has the
latest version of HyperNetX and HNXWidget installed; it also contains all the HyperNetX tutorials.

Prerequisites
*************

.. _Docker: https://docs.docker.com/engine/install/
.. _Docker-Compose: https://docs.docker.com/compose/install/

* Docker_
* Docker-Compose_

Steps
*****

#. Run the container

   #. Using Docker CLI, run the container in the foreground:

      .. code-block:: bash

        docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work hypernetx/hypernetx:latest



   #. Alternatively, create a `docker-compose.yml` file with the following:

      .. code-block:: yaml

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

    Once `docker-compose.yml` is created, run the container:

    .. code-block:: bash

        docker-compose up


#. Open Jupyter Notebook

After the container has started, access the HyperNetX Jupyter Notebooks by opening the following URL in a browser:


* http://localhost:8888/tree


