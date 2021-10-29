===============
HyperNetX (HNX)
===============

.. image:: images/hnxbasics.png
   :width: 300px
   :align: right

Description
-----------

The `HNX`_ library provides classes and methods for modeling the entities and relationships 
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
To learn more about some of our research check out our :ref:`publications`. 


For comments and questions you may contact the developers directly at: 
	hypernetx@pnnl.gov

Contents
--------

.. toctree::

   Home <self>
   overview/index
   install
   Glossary <glossary>
   core
   NWHypergraph C++ Optimization <nwhy>
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
