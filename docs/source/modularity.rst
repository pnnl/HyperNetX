.. _modularity:


=========================
Modularity and Clustering
=========================

.. image:: images/ModularityScreenShot.png
   :width: 300px
   :align: right

Overview
--------
The hypergraph_modularity submodule in HNX provided functions to compute **hypergraph modularity** for a
given partition of the vertices in a HNX hypergraph. It also provides two functions to generate such
partitions running either **Kumar's algorithm**, or a simple **Last-Step algorithm**. Finally, a function
is supplied to generate the **two-section graph** for a given hypergraph which can then be used to find
vertex partition via graph-based algorithms.


Installation
------------
As it is part of HNX, installing

    >>> pip install hypernetx

also loads this submodule. It can be imported as follows:

    >>> import hypernetx.algorithms.hypergraph_modularity as hmod

Using the Tool
--------------

Precomputation
^^^^^^^^^^^^^^
* bullet 1
  * sub
* bullet2

Modularity
^^^^^^^^^^

Two-section graph
^^^^^^^^^^^^^^^^^
  
Clustering Algorithms
^^^^^^^^^^^^^^^^^^^^^

Other Features
^^^^^^^^^^^^^^

.. _HypernetxWidget: https://github.com/pnnl/hypernetx-widget
