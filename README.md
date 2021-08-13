<img src="docs/source/images/harrypotter_basic_hyp.png" align="right" width="300pt">

HyperNetX
=========

The HNX library provides classes and methods for modeling the entities and relationships 
found in complex networks as hypergraphs, the natural models for multi-dimensional network data.
As strict generalizations of graphs, hyperedges can represent arbitrary multi-way relations 
among entities, and in particular can distinguish cliques and simplices, and admit singleton edges.
As both vertex adjacency and edge
incidence are generalized to be quantities,
hypergraph paths and walks thereby have both length and *width* because of these multiway connections. 
Most graph metrics have natural generalizations to hypergraphs, but since
hypergraphs are basically set systems, they also admit to the powerful tools of algebraic topology,
including simplicial complexes and simplicial homology, to study their structure.

This library serves as a repository of the methods and algorithms we find most useful
as we explore what hypergraphs can tell us. We have a growing community of users and contributors. 
To learn more about some of our research check out our publications below:

Publications
------------
Joslyn, Cliff A; Aksoy, Sinan; Callahan, Tiffany J; Hunter, LE; Jefferson, Brett ; Praggastis, Brenda ; Purvine, Emilie AH ; Tripodi, Ignacio J: (2020) "Hypernetwork Science: From Multidimensional Networks to Computational Topology", in: Int. Conf. Complex Systems (ICCS 2020), https://arxiv.org/abs/2003.11782, (in press)

Feng, Song; Heath, Emily; Jefferson, Brett; Joslyn, CA; Kvinge, Henry; McDermott, Jason E ; Mitchell, Hugh D ; Praggastis, Brenda ; Eisfeld, Amie J; Sims, Amy C ; Thackray, Larissa B ; Fan, Shufang ; Walters, Kevin B; Halfmann, Peter J ; Westhoff-Smith, Danielle ; Tan, Qing ; Menachery, Vineet D ; Sheahan, Timothy P ; Cockrell, 
Adam S ; Kocher, Jacob F ; Stratton, Kelly G ; Heller, Natalie C ; Bramer, Lisa M ; Diamond, Michael S ; Baric, Ralph S ; Waters, Katrina M ; Kawaoka, Yoshihiro ; Purvine, Emilie: (2020) "Hypergraph Models of Biological Networks to Identify Genes Critical to Pathogenic Viral Response", in: https://arxiv.org/abs/2010.03068, BMC Bioinformatics, 22:287, doi: 10.1186/s12859-021-04197-2

Aksoy, Sinan G; Joslyn, Cliff A; Marrero, Carlos O; Praggastis, B; Purvine, Emilie AH: (2020) "Hypernetwork Science via High-Order Hypergraph Walks", EPJ Data Science, v. 9:16, https://doi.org/10.1140/epjds/s13688-020-00231-0

Joslyn, Cliff A; Aksoy, Sinan; Arendt, Dustin; Firoz, J; Jenkins, Louis ; Praggastis, Brenda ; Purvine, Emilie AH ; Zalewski, Marcin: (2020) "Hypergraph Analytics of Domain Name System Relationships", in: 17th Wshop. on Algorithms and Models for the Web Graph (WAW 2020), Lecture Notes in Computer Science, v. 12901, ed. Kaminski, B et al., pp. 1-15, Springer, https://doi.org/10.1007/978-3-030-48478-1_1

Joslyn, Cliff A; Aksoy, Sinan; Arendt, Dustin; Jenkins, L; Praggastis, Brenda; Purvine, Emilie; Zalewski, Marcin: (2019) "High Performance Hypergraph Analytics of Domain Name System Relationships", in: Proc. HICSS Symp. on Cybersecurity Big Data Analytics, http://www.azsecure-hicss.org/

Notes
-----

HNX was developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principle Developer and Designer: Brenda Praggastis
* Visualization: Dustin Arendt, Ji Young Yun
* High Performance Computing: Tony Liu, Andrew Lumsdaine
* Principal Investigator: Cliff Joslyn
* Program Manager: Mark Raugas, Brian Kritzstein
* Contributors: Sinan Aksoy, Dustin Arendt, Cliff Joslyn, Nicholas Landry, Andrew Lumsdaine, Tony Liu, Brenda Praggastis, Emilie Purvine, Mirah Shi, Francois Theberge

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: <https://pnnl.github.io/HyperNetX/>
For questions and comments contact the developers directly at: <hypernetx@pnnl.gov>

New Features of Version 1.0
---------------------------

1. Hypergraph construction can be sped up by reading in all of the data at once. In particular the hypergraph constructor may read a Pandas dataframe object and create edges and nodes based on column headers. The new hypergraphs are given an attribute `static=True`.
2. A C++ addon called [NWHy](docs/build/nwhy.html) can be used in Linux environments to support optimized hypergraph methods such as s-centrality measures.
3. A JavaScript addon called [Hypernetx-Widget](docs/build/widget.html) can be used to interactively inspect hypergraphs in a Jupyter Notebook.
4. Four new tutorials highlighting the s-centrality metrics, static Hypergraphs, [NWHy](docs/build/nwhy.html), and [Hypernetx-Widget](docs/build/widget.html).

New Features of Version 1.1
---------------------------

1. Static Hypergraph refactored to improve performance across all methods.
2. Added modules and tutorials for Contagion Modeling, Community Detection, Clustering, and Hypergraph Generation.
3. Cell weights for incidence matrices may be added to static hypergraphs on construction.

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
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%206%20-%20Static%20Hypergraphs%20and%20Entities.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Tutorial 6 - Static Hypergraphs and Entities</span>
</a>
</br>

<a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%207%20-%20s-centrality.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    <span >Tutorial 7 - s-Centrality</span>
</a>
</br>
    
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


