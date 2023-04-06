HyperNetX
=========

The HyperNetX library provides classes and methods for the analysis
and visualization of complex network data modeled as hypergraphs.
The library generalizes traditional graph metrics.

HypernetX was developed by the Pacific Northwest National Laboratory for the
Hypernets project as part of its High Performance Data Analytics (HPDA) program.
PNNL is operated by Battelle Memorial Institute under Contract DE-ACO5-76RL01830.


- Principle Developer and Designer: Brenda Praggastis
- Lead developer for 2.0: Madelyn Shapiro
- Continuous integration and development: Mark Bonicillo
- Visualization: Dustin Arendt, Ji Young Yun
- Principal Investigator: Cliff Joslyn
- Program Manager: Brian Kritzstein
- Contributors: Sinan Aksoy, Dustin Arendt, Mark Bonicillo, Cliff Joslyn, Nicholas Landry, Andrew Lumsdaine, Tony Liu, Audun Meyers, Christopher Potvin, Brenda Praggastis, Emilie Purvine, Madelyn Shapiro, Mirah Shi, François Théberge

The code in this repository is intended to support researchers modeling data
as hypergraphs. We have a growing community of users and contributors.
Documentation is available at: https://pnnl.github.io/HyperNetX

For questions and comments contact the developers directly at: hypernetx@pnnl.gov

**New Features in Version 2.0**
===============================

HNX 2.0 now accepts metadata as core attributes of the edges and nodes of a 
hypergraph. While the library continues to accept lists, dictionaries and 
dataframes as basic inputs for hypergraph constructions, both cell 
properties and edge and node properties can now be easily added for 
retrieval as object attributes. The core library has been rebuilt to take 
advantage of the flexibility and speed of Pandas Dataframes.
Dataframes offer the ability to store and easily access hypergraph metadata.
Metadata can be used for filtering objects, and characterize their 
distributions by their attributes.
Version 2.0 is not backwards compatible. Objects constructed using version 
1.x can be imported from their incidence dictionaries. 

New features to look for:
~~~~~~~~~~~~~~~~~~~~~~~~~

1. 	Hypergraph constructor accepts cell properties and object properties as 	dict or pd.DataFrame. 
2. 	Cell weights now available in the incidence matrix (fixes bug in earlier 
	release).
3. 	API does not require user to access the Entity or EntitySet classes. 	
	Instead all user input is inserted directly into the hypergraph 
	constructor and user defined ids for edges and nodes do not have to be disjoint sets.
4.	New tutorials for barycentric homology and zigzag homology have been 
	added.
5.  The Modularity tutorial was updated - removing bugs.
6.  NWHy is no longer supported as new Pandas backend has provided better 
	scalability.
7.	No distinction between dynamic and static hypergraph is made. ADD/REMOVE 
	functionality will be in V. 2.1. 




