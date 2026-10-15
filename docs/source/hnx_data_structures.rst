
.. _hypconstructors:


===================
HNX Data Structures
===================

..  figure:: images/code_structure.png
   :width: 300px
   :align: right
   
   Code structure for HNX.

The HNX library centers around the idea of a :term:`hypergraph`.  
There are many definitions of a *hypergraph*. In HNX a hypergraph
is a tuple of three sets, :math:`H =  (V, E, \mathcal{I})`. 

- :math:`V`, a set of *nodes* (aka hypernodes, vertices), distinguished by unique identifiers
- :math:`E` a set of *edges* (aka hyperedges), distinguished by  unique identifiers
- :math:`\mathcal{I}`, a set of *incidences*, which form a subset of :math:`E \times V`, distinguished by the pairing of unique identifiers of edges in :math:`E` and nodes in :math:`V`

The incidences :math:`\mathcal{I}` can be described by a Boolean function, :math:`\mathcal{I}_B : E \times V \rightarrow \{0, 1\}`, indicating whether or not a pair is included in the hypergraph.

In HNX we instantiate :math:`H =  (V, E, \mathcal{I})` using three *hypergraph views.* We can visualize this through a high 
level diagram of our current code structure shown in Fig. 1. Here we begin with data (e.g., data frame, dictionary, 
list of lists, etc.) that is digested via the appropriate factory method to construct property stores for nodes, 
edges, and incidences as well as an incidence store that captures the hypergraph structure. 
These four objects are then used to create three hypergraph views that the hypergraph object 
uses to access and analyze the hypergraph structure and attributes.

