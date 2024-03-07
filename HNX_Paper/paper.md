---
title: 'HyperNetX: A Python package for modeling complex network data as hypergraphs'
tags:
- Python
- hypergraph
- network science
- simplicial-complexes
- knowledge graph
- simplicial-homology
- s-linegraph
- property hypergraph
authors:
- name: Brenda Praggastis
  orcid: 0000-0003-1344-0497
  affiliation: "1"
- name: Sinan Aksoy
  orcid: 0000-0002-3466-3334
  affiliation: "1"
- name: Dustin Arendt
  orcid: 0000-0003-2466-199X
  affiliation: "1"
- name: Mark Bonicillo
  affiliation: "1"
- name: Cliff Joslyn
  orcid: 0000-0002-5923-5547
  affiliation: "1"
- name: Emilie Purvine
  orcid: 0000-0003-2069-5594
  affiliation: "1"
- name: Madelyn Shapiro
  orcid: 0000-0002-2786-7056
  affiliation: "1"
- name: Ji Young Yun
  affiliation: "1"
affiliations:
- name: Pacific Northwest National Laboratory, USA
  index: 1

date: 21 June 2023
bibliography: paper.bib

---

# Summary
HyperNetX (HNX) is an open source Python library for the analysis and visualization of complex network data modeled as hypergraphs.
Initially released in 2019, HNX facilitates exploratory data analysis of complex networks using algebraic topology, combinatorics, and generalized hypergraph and graph theoretical methods on structured data inputs.
With its 2023 release, the library supports attaching metadata, numerical and categorical, to nodes (vertices) and  hyperedges, as well as to node-hyperedge pairings (incidences).
HNX has a customizable Matplotlib-based visualization module as well as HypernetX-Widget, its JavaScript addon for interactive exploration and visualization of hypergraphs within Jupyter Notebooks. Both packages are available on GitHub and PyPI. With a growing community of users and collaborators, HNX has become a preeminent tool for hypergraph analysis.

![HNX-Widget is an add-on for the Jupyter Notebook
computational environment, enabling users to view and interactively
explore hypergraphs.
The main features of the tool are: 1) adjustable layout 2) advanced
selection and 3) visual encoding of node and edge properties.
Metadata may be attached to the tool by providing tabular data via two optional data frames indexed by node and hyperedge identifiers. Above is an HNX-Widget visualization of a Scene to Character mapping from the LesMis dataset [@knuth1993].](Figures/hnxexample.png){height="225pt"}


# Statement of need
For more than a century, graph theory has provided powerful methods for studying network relationships among abstract entities.
Since the early 2000's, software packages such as NetworkX [@SciPyProceedings_11] and `igraph` [@csardi2006igraph;@antonov2023igraph]  have
made these theoretical tools available to data scientists for studying large data sets.
Graphs represent pairwise interactions between entities, but for many network datasets this is a severe limitation.
In 1973, hypergraphs were introduced by Claude Berge [@Berge1973Graphs] as a strict generalization of graphs: a hyperedge in a hypergraph can contain any number of nodes, including 1, 2, or more.
Hypergraphs have been used to model complex network datasets in
areas such as the biological sciences, information systems, the power grid, and cyber security.
Hypergraphs strictly generalize graphs (all graphs are (2-uniform) hypergraphs), and thus can represent additional data complexity and have more mathematical properties to exploit (for example, hyperedges can be contained in other hyperedges). As mathematical set systems, simplicial and homological methods from
Algebraic Topology are well suited to aid in their analysis [@Joslyn2021;@Torres2021].
With the development of hypergraph modeling methods, new software was required to support
experimentation and exploration, which prompted the development of HyperNetX.

## Related Software
Due to the diversity of hypergraph modeling applications, hypergraph software libraries are
often bootstrapped using data structures and methods most appropriate to their usage.
In 2020 SimpleHypergraph.jl  was made available for high performance computing on hypergraphs using Julia.
The library offers a suite of tools for centrality analysis and community detection and integrates its own
visualization tools with those offered by HNX [@Szufel2019]. In 2021 CompleX Group Interactions (XGI)  was released.
Originally developed to efficiently discover spreading processes in complex social systems, the library now offers
a statistics package as well as a full suite of hypergraph analysis and visualization tools [@Landry2023].
More recently, in 2023 HyperGraphX (HGX)  was released, again with a full suite of tools for community detection
as well as general hypergraph analytics [@Lotito2023Hypergraphx].
A nice compendium of many of the hypergraph libraries created in the last decade can be found in @Kurte2021.

HNX leads the effort to share library capabilities by specifying a Hypergraph Interchange Format (HIF)
for storing hypergraph data as a JSON object. Since hypergraphs can store metadata on its nodes,
hyperedges, and incidence pairs, a standardized format makes it easy to share hypergraphs across libraries.

![Visualizations from hypergraph libraries based on the bipartite graph seen in grey
  under the HyperNetX visualization (left side): XGI (Center), @Landry2023 and SimpleHypergraph (Right), @Szufel2019.](Figures/3graphs.png)

# Overview of HNX
HNX serves as a platform for the collaboration and sharing of hypergraph
methods within the research community.
Originally intended to generalize many of the methods from NetworkX
to hypergraphs, HNX now has implementations for many hypergraph-specific metrics.
While graph paths can be measured by length,
hypergraph paths also have a width parameter *s*, given by the minimum intersection size
of incident hyperedges in the path [@Aksoy2020Hypernetwork].
HNX uses this *s* parameter in many of
its core methods as well as in its *s-*centrality module.
As set systems, hypergraphs can be viewed as subsets of abstract simplicial
complexes – combinatorial projections of geometric objects constructed from points, line
segments, triangles, tetrahedrons, and their higher dimensional analogues.
HNX's Simplicial Homology module identifies and computes the *voids* of different dimensions
in the simplicial complexes generated by modestly sized hypergraphs.
These objects, which are used for defining the *Homology Groups*
studied by Algebraic Topologists, offer new metrics for exploratory
data science.

As a collaborative platform, HNX contains contributed modules
and tutorials in the form of Jupyter notebooks
for Laplacian clustering, clustering and modularity, synthetic
generation of hypergraphs, and Contagion Theory.
In its latest release, HNX 2.0 uses Pandas dataframes [@reback2020pandas;@mckinney-proc-scipy-2010] as its underlying data structure,
making the nodes and hyperedges of a hypergraph as accessible as the
cells in a dataframe.
This simple design allows HNX to import data from semantically
loaded graphs such as property graphs and knowledge graphs,
in order to model and explore their higher order relationships.
Because it is open source, HNX provides a unique opportunity for
hypergraph researchers to implement their own methods built from
HNX and contribute them as modules and Jupyter tutorials to the HNX user community.

## Projects using HNX
HNX was created by the Pacific Northwest National Laboratory. It has provided data analysis and visualization support for academic papers in subject areas such as biological systems [@Feng2021;@Colby2023], cyber security [@Joslyn2020DNS], information systems [@Molnar2022Application], neural networks [@Praggastis2022SVD], knowledge graphs [@joslyn2018], and the foundations of hypergraph theory [@Vazquez2022Growth].

# References
