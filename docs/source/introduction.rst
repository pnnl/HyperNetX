.. _introduction:

********************
Introduction
********************


This documentation provides a comprehensive guide to the HyperNetworkX (HNX) library, a Python package for working with hypergraphs. This guide covers the mathematical foundations of hypergraphs, the key terminology used within HNX, the core data structures employed by the library, and practical methods for constructing hypergraphs. Specifically:

*   **A Gentle Introduction to Hypergraph Mathematics:** This section provides a basic introduction to hypergraphs, explaining how they generalize traditional graphs and introducing key concepts like hyperedges, incidence matrices, and the duality of hypergraphs. It focuses on "gentle" hypergraphs (simple, finite, connected, etc.) to build a solid foundation before touching on more complex topics.

*   **Glossary of HNX Terms:** This section offers a comprehensive glossary of terms used throughout the HNX library documentation. It defines key concepts related to hypergraphs, such as nodes, edges, incidences, various types of adjacency, connectivity, and distance, as well as HNX-specific terms like PropertyStore, IncidenceStore, and HypergraphView.

*   **Data Structures:** This section describes the core data structures used within the HNX library to represent and manage hypergraphs. It explains how these structures organize nodes, edges, and incidences, and how they facilitate efficient access and manipulation of hypergraph data.

*   **Hypergraph Constructors:** This section details the various ways to construct hypergraphs using the HNX library. It explains how to create hypergraphs from different data structures, including lists of lists, dictionaries, and Pandas DataFrames, and describes how to associate metadata and properties with nodes, edges, and incidences.

For more detailed information on each of these areas, please explore the following sections:

.. toctree::
   :maxdepth: 2

   A Gentle Introduction to Hypergraph Mathematics <hypergraph101>
   Glossary <glossary>
   Data Structures <hnx_data_structures>
   Hypergraph Constructors <hypconstructors>