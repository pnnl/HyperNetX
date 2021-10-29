.. overview:

========
Overview
========

.. image:: ../images/harrypotter_basic_hyp.png
   :width: 300px
   :align: right

The `HyperNetX`_ (`HNX`_) library was developed to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
For questions and comments you may contact the developers directly at: hypernetx@pnnl.gov

`HyperNetX`_ was developed by the `Pacific Northwest National Laboratory <https://www.pnnl.gov/>`_ for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principle Developer and Designer: Brenda Praggastis
* Visualization: Dustin Arendt, Ji Young Yun
* High Performance Computing: Tony Liu, Andrew Lumsdaine
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Mathematics, methods, and algorithms: Sinan Aksoy, Dustin Arendt, Cliff Joslyn, Nicholas Landry, Tony Liu, Andrew Lumsdaine, Brenda Praggastis, and Emilie Purvine, François Théberge



New Features in Version 1.0
---------------------------

#. Hypergraph construction can be sped up by reading in all of the data at once. In particular the hypergraph constructor may read a Pandas dataframe object and create edges and nodes based on column headers. 
#. The C++ addon :ref:`nwhy` can be used in Linux environments to support optimized hypergraph methods such as s-centrality measures.
#. The JavaScript addon :ref:`widget` can be used to interactively inspect hypergraphs in a Jupyter Notebook.
#. We've added four new tutorials highlighting the s-centrality metrics, static Hypergraphs, :ref:`nwhy`, and :ref:`widget`.

New Features in Version 1.1
---------------------------

#. Cell weights for incidence matrices.
#. Support for edge and node properties in static hypergraphs.
#. Three new algorithms modules and their corresponding tutorials

   #. Contagion module for studying SIS and SIR contagion networks using hypergraphs.
   #. Clustering module for clustering vertices based on hyperedge incidence and weighting.
   #. Generator module for synthetic generation of ChungLu and DCSBM hypergraphs.
   
New Features in Version 1.2
---------------------------
#. Added algorithm module and tutorial for Modularity and Clustering


.. _colab:

COLAB Tutorials
---------------
The following tutorials may be run in your browser using Google Colab. Additional tutorials are
available on `GitHub <https://github.com/pnnl/HyperNetX>`_.

.. raw:: html

   <div>
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
   <a href="https://colab.research.google.com/github/pnnl/HyperNetX/blob/master/tutorials/Tutorial%20%20-%20s-centrality.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
      <span >Tutorial 7 - s-Centrality</span>
   </a>
   </br></br></br>
   </div>


Notice
------
This material was prepared as an account of work sponsored by an agency of the United States Government.  
Neither the United States Government nor the United States Department of Energy, nor Battelle, 
nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of 
these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility 
for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process 
disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, 
trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, 
or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. 
The views and opinions of authors expressed herein do not necessarily state or reflect 
those of the United States Government or any agency thereof.


.. raw:: html

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
HyperNetX is released under the 3-Clause BSD license (see :ref:`license`)

.. toctree::
   :maxdepth: 2


.. _HyperNetX: https://github.com/pnnl/HyperNetX
.. _HNX: https://github.com/pnnl/HyperNetX
