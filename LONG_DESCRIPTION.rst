.. _long_description:

HyperNetX
=========

The HyperNetX (HNX) library provides classes and methods for the analysis
and visualization of complex network data modeled as hypergraphs.
The library generalizes traditional graph metrics.
Documentation for HNX is available at: https://hypernetx.readthedocs.io/

HNX was originally developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
It is currently maintained by scientists at PNNL, but contributions and bug fixes
from the community are welcome and encouraged.
Please see our [Contributor's Guide](https://hypernetx.readthedocs.io/en/latest/contributions.html)
for more information.

PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.

* Principal Developer and Designer: Brenda Praggastis
* Development Team: Brenda Praggastis, Audun Myers, Greg Roek, Ryan Danehy
* Visualization: Dustin Arendt, Ji Young Yun
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Principal Contributors (Design, Theory, Code): Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Ryan Danehy, Helen Jenne, Cliff Joslyn, Nicholas Landry, Audun Myers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Greg Roek, Mirah Shi, Francois Theberge, Ji Young Yun

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
HNX is a primary contributor to the
Hypergraph Interchange Format (HIF), a json schema for sharing data
modeled as hypergraphs. The specification and sample notebooks may be found
here: https://github.com/pszufe/HIF-standard/tree/main
Other hypergraph libraries using this standard are listed below:

- [HypergraphX (HGX)](https://github.com/HGX-Team/hypergraphx) (Python)
- [CompleX Group Interactions (XGI)](https://github.com/xgi-org/xgi) (Python)
- [SimpleHypergraphs.jl](https://github.com/pszufe/SimpleHypergraphs.jl) (Julia)
- [Hypergraph-Analysis-Toolbox(HAT)](https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox) (Python)

For questions and comments about HNX contact the developers directly at: hypernetx@pnnl.gov.

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
