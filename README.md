<img src="docs/source/images/harrypotter_basic_hyp.png" align="right" width="300pt">

HyperNetX
=========

The HyperNetX library provides classes and methods for the analysis
and visualization of complex network data modeled as hypergraphs. The library generalizes traditional graph metrics.

HypernetX was developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principle Developer and Designer: Brenda Praggastis
* Visualization: Dustin Arendt, Ji Young Yun
* High Performance Computing: Tony Liu, Andrew Lumsdaine
* Principal Investigator: Cliff Joslyn
* Program Manager: Mark Raugas, Brian Kritzstein
* Mathematics, methods, and algorithms: Sinan Aksoy, Dustin Arendt, Cliff Joslyn, Andrew Lumsdaine, Tony Liu, Brenda Praggastis, and Emilie Purvine

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: <https://pnnl.github.io/HyperNetX/>
For questions and comments contact the developers directly at:
	<hypernetx@pnnl.gov>

New Features of Version 1.0:

1. Hypergraph construction can be sped up by reading in all of the data at once. In particular the hypergraph constructor may read a Pandas dataframe object and create edges and nodes based on column headers. 
2. A C++ addon called [NWHy](docs/build/nwhy.html) can be used in Linux environments to support optimized hypergraph methods such as s-centrality measures.
3. A JavaScript addon called [Hypernetx-Widget](docs/build/widget.html) can be used to interactively inspect hypergraphs in a Jupyter Notebook.
4. We've added three new tutorials. One highlights the s-centrality metrics. The other two introduce Static Hypergraphs and NWHy.

Tutorials may be run in your browser using Google Colab
-------------------------------------------------------

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%201%20-%20HNX%20Basics.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 1 - HNX Basics</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%202%20-%20Visualization%20Methods.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 2 - Visualization Methods</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%203%20-%20LesMis%20Case%20Study.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 3 - LesMis Case Study</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%204%20-%20LesMis%20Visualizations-BookTour.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 4 - LesMis Visualizations-Book Tour</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%205%20-%20Homology%20mod%202%20for%20TriLoop%20Example.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 5 - Homology mod2 for TriLoop Example</span>
</a>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%206%20-%20Static%20Hypergraphs%20and%20Entities.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 6 - Static Hypergraphs and Entities</span>
</a>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%20%20-%20s-centrality.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
	<span >Tutorial 7 - s-Centrality</span>
</a>

	
Installing HyperNetX
====================
HyperNetX may be cloned or forked from: <https://github.com/pnnl/HyperNetX> 

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

To use [NWHy](docs/build/nwhy.html) use python=3.9 and the conda version of tbb in your environment. 
**Note** that [NWHy](docs/build/nwhy.html) only works on Linux and some OSX systems. See [NWHy documentation](docs/build/nwhy.html) for more.:

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

Notice
------
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

   <div align=center>
   <pre style="align-text:center;font-size:10pt">
   PACIFIC NORTHWEST NATIONAL LABORATORY
   operated by
   BATTELLE
   for the
   UNITED STATES DEPARTMENT OF ENERGY
   under Contract DE-AC05-76RL01830
   </pre>
   </div>


License
-------

Released under the 3-Clause BSD license (see License.rst)


