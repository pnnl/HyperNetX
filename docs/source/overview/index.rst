.. _overview:

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

* Principal Developer and Designer: Brenda Praggastis
* Development Team: Madelyn Shapiro, Mark Bonicillo
* Visualization: Dustin Arendt, Ji Young Yun
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Principal Contributors (Design, Theory, Code): Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Helen Jenne, Cliff Joslyn, Nicholas Landry, Audun Myers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Greg Roek, Madelyn Shapiro, Mirah Shi, Francois Theberge, Ji Young Yun



New Features in Version 2.0
---------------------------

HNX 2.0 now accepts metadata as core attributes of the edges and nodes of a
hypergraph. While the library continues to accept lists, dictionaries and
dataframes as basic inputs for hypergraph constructions, both cell
properties and edge and node properties can now be easily added for
retrieval as object attributes. The core library has been rebuilt to take
advantage of the flexibility and speed of Pandas Dataframes.
Dataframes offer the ability to store and easily access hypergraph metadata.
Metadata can be used for filtering objects, and characterize their
distributions by their attributes.
**Version 2.0 is not backwards compatible. Objects constructed using version
1.x can be imported from their incidence dictionaries.**

New features to look for:
~~~~~~~~~~~~~~~~~~~~~~~~~

#. The Hypergraph constructor now accepts nested dictionaries with incidence cell properties, pandas.DataFrames, and 2-column Numpy arrays.
#. Additional constructors accept incidence matrices and incidence dataframes.
#. Hypergraph constructors accept cell, edge, and node metadata. 
#. Metadata available as attributes on the cells, edges, and nodes. 
#. User defined cell weights and default weights available to incidence matrix.
#. Meta data persists with restrictions and removals.
#. Meta data persists onto s-linegraphs as node attributes of Networkx graphs.
#. New module and tutorial for *Barycentric homology*
#. New hnxwidget available using  `pipinstall hnxwidget`.
#. The `static` and `dynamic` distinctions no longer exist. All hypergraphs use the same underlying data structure, supported by Pandas dataFrames. All hypergraphs maintain a `state_dict` to avoid repeating computations.
#. Methods for adding nodes and hyperedges are currently not supported. 
#. Methods for removing nodes return new hypergraph.
#. The `nwhy` optimizations are no longer supported.
#. Entity and EntitySet classes are being moved to the background. The Hypergraph constructor does not accept either. 


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
