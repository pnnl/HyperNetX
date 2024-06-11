.. _long_description:

HyperNetX
=========

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

HyperNetX 2.3
~~~~~~~~~~~~~

HyperNetX 2.3. is the latest, stable release. The core library has been refactored to take better advantage
of Pandas Dataframes, improve readability and maintainability, address bugs, and make it easier to change.
New features have been added, most notably the ability to add and remove edges, nodes, and incidences. Updating is recommended.

**Version 2.3 is not backwards compatible. Objects constructed using earlier versions
can be imported using their incidence dictionaries and/or property datafames.**

What's New
~~~~~~~~~~~~~~~~~~~~~~~~~
#. We've added new functionality to Hypergraphs; you can add and remove nodes, edges, and incidences on Hypergraph.
#. Arithmetic operations have also been added to Hypergraph: sum, difference, union, intersection.
#. We've also added a new tutorial on basic hypergraph arithmetic operations.
#. Under the hood, the EntitySet has been replaced by HypergraphView, new factory methods have been created to support the Hypergraph constructor, and internal classes such as IncidenceStore and PropertyStore help maintain the structure and attributes of a Hypergraph.

What's Changed
~~~~~~~~~~~~~~~~~~~~~~~~~
#. Documentation has received a major update; the Glossary and docstrings of Hypergraph have been updated.
#. HNX now requires Python >=3.10,<4.0.0
#. We've upgraded all the underlying core libraries to the latest versions.
