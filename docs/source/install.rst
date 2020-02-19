Installing HyperNetX
====================

HyperNetX may be cloned or forked from: https://github.com/pnnl/HyperNetX .

To install in an Anaconda environment
-------------------------------------

	>>> conda create -n <env name> python=3.6
	>>> source activate <env name> 

Mac Users: If you wish to build the documentation you will need
the conda version of matplotlib:
	
	>>> conda install matplotlib

To install in a virtualenv environment
--------------------------------------

	>>> virtualenv --python=<path to python 3.6 executable> <path to env name>

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


Install using Pip
-----------------
For a minimal installation:

	>>> pip install hypernetx

For an editable installation with access to jupyter notebooks: 

    >>> pip install [-e] .

To install with the tutorials: 

	>>> pip install -e .['tutorials']

To install with the documentation: 
	
	>>> pip install -e .['documentation']
	>>> chmod 755 build_docs.sh
	>>> ./build_docs.sh
	## This will generate the documentation in /docs/build
	## Open them in your browser with /docs/build/index.html

To install and test using pytest:

	>>> pip install -e .['testing']
	>>> pytest

To install the whole shabang:

	>>> pip install -e .['all']


License
-------

Released under the 3-Clause BSD license (see :ref:`license`)


