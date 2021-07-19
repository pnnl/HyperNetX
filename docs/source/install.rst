Installing HyperNetX
====================

HyperNetX may be cloned or forked from: https://github.com/pnnl/HyperNetX .

To install in an Anaconda environment
-------------------------------------

	>>> conda create -n <env name> python=3.7
	>>> source activate <env name>
	>>> pip install hypernetx

Mac Users: If you wish to build the documentation you will need
the conda version of matplotlib:

	>>> conda create -n <env name> python=3.7 matplotlib
	>>> source activate <env name>
	>>> pip install hypernetx	

To use :ref:`NWHy <nwhy>` use python=3.9 and the conda version of tbb in your environment. 
**Note** that :ref:`NWHy <nwhy>` only works on Linux and some OSX systems. See NWHy docs for more.:

	>>> conda create -n <env name> python=3.9 tbb
	>>> source activate <env name>
	>>> pip install hypernetx
	>>> pip install nwhy

To install in a virtualenv environment
--------------------------------------

	>>> virtualenv --python=<path to python 3.7 executable> <path to env name>

This will create a virtual environment in the specified location using
the specified python executable. For example:

	>>> virtualenv --python=C:\Anaconda3\python.exe hnx

This will create a virtual environment in .\hnx using the python
that comes with Anaconda3.

	>>> <path to env name>\Scripts\activate<file extension>

If you are running in Windows PowerShell use <file extension>=.ps1

If you are running in Windows Command Prompt use <file extension>=.bat

Otherwise use <file extension>=NULL (no file extension).

Once activated continue to follow the installation instructions below.


Install using Pip options
-------------------------
For a minimal installation:

	>>> pip install hypernetx

For an editable installation with access to jupyter notebooks:

    >>> pip install [-e] .

To install with the tutorials:

	>>> pip install -e .['tutorials']

To install with the documentation:

	>>> pip install -e .['documentation']
	>>> chmod 755 build_docs.sh
	>>> sh build_docs.sh
	## This will generate the documentation in /docs/build/
	## Open them in your browser with /docs/index.html

To install and test using pytest:

	>>> pip install -e .['testing']
	>>> pytest

To install the whole shabang:

	>>> pip install -e .['all']






