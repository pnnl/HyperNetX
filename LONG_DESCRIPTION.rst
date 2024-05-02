.. _long_description:

HyperNetX
=================

The HyperNetX library provides classes and methods for the analysis
and visualization of complex network data modeled as hypergraphs.
The library generalizes traditional graph metrics.

HypernetX was developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principal Developer and Designer: Brenda Praggastis
* Development Team: Audun Myers, Mark Bonicillo
* Visualization: Dustin Arendt, Ji Young Yun
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Principal Contributors (Design, Theory, Code): Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Helen Jenne, Cliff Joslyn, Nicholas Landry, Audun Myers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Greg Roek, Mirah Shi, Francois Theberge, Ji Young Yun

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: https://pnnl.github.io/HyperNetX

For questions and comments contact the developers directly at: hypernetx@pnnl.gov

Summary - Release highlights - HNX 2.3
--------------------------------------

HyperNetX 2.3. is the latest, stable release. The core library has been refactored to take better advantage
of Pandas Dataframes, improve readability and maintainability, address bugs, and make it easier to change.
New features have been added, most notably the ability to add and remove edges, nodes, and incidences.

**Version ^2.0 is not backwards compatible. Objects constructed using version
1.x can be imported from their incidence dictionaries.**

What's New
~~~~~~~~~~~~~~~~~~~~~~~~~
#. Hypergraph now supports adding and removing edges, nodes, and incidences
#. Hypergraph also supports the sum, difference, union, and intersection of a Hypergraph to another Hypergraph
#. New factory methods to support the Hypergraph constructor
#. EntitySet has been replaced by HypergraphView
#. IncidenceStore and PropertyStore are new classes that maintain the structure and attributes of a Hypergraph
#. Hypergraph constructors accept cell, edge, and node metadata.


What's Changed
~~~~~~~~~~~~~~~~~~~~~~~~~
#. HNX now requires Python ^3.10,<=3.12
#. HNX core libraries have been updated
#. Updated tutorials
