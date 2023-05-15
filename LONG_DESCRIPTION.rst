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
* Development Team: Madelyn Shapiro, Mark Bonicillo
* Visualization: Dustin Arendt, Ji Young Yun
* Principal Investigator: Cliff Joslyn
* Program Manager: Brian Kritzstein
* Principal Contributors (Design, Theory, Code): Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Helen Jenne, Cliff Joslyn, Nicholas Landry, Audun Myers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Greg Roek, Madelyn Shapiro, Mirah Shi, Francois Theberge, Ji Young Yun

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: https://pnnl.github.io/HyperNetX

For questions and comments contact the developers directly at: hypernetx@pnnl.gov

New Features in Version 2.0
---------------------------

HNX 2.0 now accepts metadata as core attributes of the edges and nodes of a
hypergraph. While the library continues to accept lists, dictionaries and
dataframes as basic inputs for hypergraph constructions, both cell
properties and edge and node properties can now be easily added for
retrieval as object attributes.

The core library has been rebuilt to take advantage of the flexibility and speed of Pandas Dataframes.
Dataframes offer the ability to store and easily access hypergraph metadata. Metadata can be used for filtering objects, and characterize their
distributions by their attributes.

**Version 2.0 is not backwards compatible. Objects constructed using version
1.x can be imported from their incidence dictionaries.**

What's New
~~~~~~~~~~~~~~~~~~~~~~~~~
#. The Hypergraph constructor now accepts nested dictionaries with incidence cell properties, pandas.DataFrames, and 2-column Numpy arrays.
#. Additional constructors accept incidence matrices and incidence dataframes.
#. Hypergraph constructors accept cell, edge, and node metadata.
#. Metadata available as attributes on the cells, edges, and nodes.
#. User-defined cell weights and default weights available to incidence matrix.
#. Meta data persists with restrictions and removals.
#. Meta data persists onto s-linegraphs as node attributes of Networkx graphs.
#. New hnxwidget available using  `pip install hnxwidget`.


What's Changed
~~~~~~~~~~~~~~~~~~~~~~~~~
#. The `static` and `dynamic` distinctions no longer exist. All hypergraphs use the same underlying data structure, supported by Pandas dataFrames. All hypergraphs maintain a `state_dict` to avoid repeating computations.
#. Methods for adding nodes and hyperedges are currently not supported.
#. The `nwhy` optimizations are no longer supported.
#. Entity and EntitySet classes are being moved to the background. The Hypergraph constructor does not accept either.
