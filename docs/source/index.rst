===============
HyperNetX (HNX)
===============

.. image:: images/hnxbasics.png
   :width: 300px
   :align: right


`HNX`_ is a Python library for hypergraphs, the natural models for multi-dimensional network data.  

To get started, try the  `interactive COLAB tutorials <https://pnnl.github.io/HyperNetX/build/overview/index.html#colab-tutorials>`_.  For a primer on hypergraphs, try this :ref:`gentle introduction<hypergraph101>`.  To see hypergraphs at work in cutting-edge research, see our list of recent :ref:`publications<publications>`.

Why hypergraphs?
----------------

Like graphs, hypergraphs capture important information about networks and relationships.  But hypergraphs do more -- they model *multi-way* relationships, where ordinary graphs only capture two-way relationships. This library serves as a repository of the methods and algorithms we find most useful
as we explore what hypergraphs can tell us. 

.. As both vertex adjacency and edge
.. incidence are generalized to be quantities,
.. hypergraph paths and walks thereby have both length and *width* because of these multiway connections. 
.. Most graph metrics have natural generalizations to hypergraphs, but since
.. hypergraphs are basically set systems, they also admit to the powerful tools of algebraic topology,
.. including simplicial complexes and simplicial homology, to study their structure.


Our community
-------------

We have a growing community of users and contributors. For the latest software updates, and to learn about the development team, see the library :ref:`overview<overview>`.  

Questions and comments are welcome! Contact us at
	hypernetx@pnnl.gov






Contents
--------

.. toctree::

   Home <self>
   overview/index
   install
   Glossary <glossary>
   core
   A Gentle Introduction to Hypergraph Mathematics <hypergraph101>
   HyperNetX Visualization Widget <widget>
   Algorithms: Modularity and Clustering <modularity>
   Publications <publications>
   license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _HNX: https://github.com/pnnl/HyperNetX
