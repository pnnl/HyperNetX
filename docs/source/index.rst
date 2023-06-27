===============
HyperNetX (HNX)
===============

.. image:: images/hnxbasics.png
   :width: 300px
   :align: right


`HNX`_ is a Python library for hypergraphs, the natural models for multi-dimensional network data.  

To get started, try the  :ref:`interactive COLAB tutorials<colab>`.  For a primer on hypergraphs, try this :ref:`gentle introduction<hypergraph101>`.  To see hypergraphs at work in cutting-edge research, see our list of recent :ref:`publications<publications>`.

Why hypergraphs?
----------------

Like graphs, hypergraphs capture important information about networks and relationships.  But hypergraphs do more -- they model *multi-way* relationships, where ordinary graphs only capture two-way relationships. This library serves as a repository of methods and algorithms that have proven useful over years of exploration into what hypergraphs can tell us. 

As both vertex adjacency and edge
incidence are generalized to be quantities,
hypergraph paths and walks have both length and *width* because of these multiway connections. 
Most graph metrics have natural generalizations to hypergraphs, but since
hypergraphs are basically set systems, they also admit to the powerful tools of algebraic topology,
including simplicial complexes and simplicial homology, to study their structure.


Our community
-------------

We have a growing community of users and contributors. For the latest software updates, and to learn about the development team, see the  :ref:`library overview<overview>`.   Have ideas to share?  We'd love to hear from you!  Our `orientation for contributors <https://github.com/pnnl/HyperNetX/blob/master/CONTRIBUTING.md>`_ can help you get started.

Our values
-------------

Our shared values as software developers guide us in our day-to-day interactions and decision-making. Our open source projects are no exception. Trust, respect, collaboration and transparency are core values we believe should live and breathe within our projects. Our community welcomes participants from around the world with different experiences, unique perspectives, and great ideas to share.  See our `code of conduct <https://github.com/pnnl/HyperNetX/blob/master/CODE_OF_CONDUCT.md>`_ to learn more.

Contact us
----------

Questions and comments are welcome! Contact us at
	hypernetx@pnnl.gov

Contents
--------

.. toctree::
   :maxdepth: 1

   Home <self>
   overview/index
   install
   Glossary <glossary>
   core
   A Gentle Introduction to Hypergraph Mathematics <hypergraph101>
   Hypergraph Constructors <hypconstructors>
   Visualization Widget <widget>
   Algorithms: Modularity and Clustering <modularity>
   Publications <publications>
   license
   long_description


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _HNX: https://github.com/pnnl/HyperNetX
