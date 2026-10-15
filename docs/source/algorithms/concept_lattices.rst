===========================
Hypergraph Concept Lattices
===========================

Overview
--------
The concept_lattices submodule in HNX provides functions to compute the **concept lattice** from a hypergraph's incidence relation. This code is based on the Concepts python package by Sebastian Bank: https://github.com/xflr6/concepts. 


Installation
------------
Since it is part of HNX, no extra installation is required.
The submodule can be imported as follows::

   import hypernetx.algorithms.concept_lattices as hlat

Using the Tool
--------------

Concept Lattices
^^^^^^^^^^^^^^^^

Formal concept analysis [2] is a lattice theoretic tool for studying binary relations between *objects* and their *properties*. For hypergraphs, we apply this tool to the incidence relation so that vertices correspond to objects and hyperedges to properties. Given a subset of vertices :math:`A \subseteq V` we may obtain a subset of hyperedges :math:`A' := \{ e \in E \mid \forall v \in A, \, v \in e \}` where . On the other hand, given a subset of hyperedges :math:`B \subseteq E` we may obtain a subset of vertices :math:`B' := \{ v \in V \mid \forall e \in B, \, v \in e\}`. A pair of subsets :math:`(A,B)` is a *concept* if :math:`A' = B` and :math:`B' = A`. We call A the extent and B intent of a concept. 

The lattice object can be constructed as::

    lat =  hlat.HypergraphLattice(hypergraph)

which is populated with hlat.Concept objects. These store both the extent and intent of a cocnept as well as their heirarchical strcuture. 

Lattice Operations
^^^^^^^^^^^^^^^^^^

The following lattice-theoretic operations are implemented as well: join (:math:`\vee` least upper bound), meet (:math:`\wedge` greatest lower bound), upsets, downsets, atoms (minimal elements that are **not** the bottom), and join irreducibles (elements which are not the join of two smaller elements or the bottom). 
They are computed as follows::

    lat.join( [concept1, concept2] ) 
    lat.meet( [concept1, concept2] )
    lat.upset( concept )
    lat.downset( concept )
    lat.atoms()
    lat.join_irreducibles()

Ploting
^^^^^^^

We provide a basic plotting function using networkx's drawing functions to display the lattice as a directed graph. Concepts :math:`(A,B) < (C,D)`
with no other concepts in between them are connected by a directed edge in the plot. To plot, one may run for example::

    hg_lat.draw_lattice(concept_labels= True, shortform = False)

which displays the lattice along with the full concept labellings. For truncated labellings, set shortform = True, which displays objects/attributes added to the concept that were not present in smaller/larger concepts respectively. 

Lattice Metrics
^^^^^^^^^^^^^^^

We supply two types of metrics on lattices: the shortest path metric on the Hasse diagram and a lattice valuation based metric from [1]. One can compute the distance between two concepts via::

    lat.distance(x,y, metric)

where x and y are either indices of concepts or the concept objects themselves. The default metric is the shortest path distance on the Hasse diagram of the lattice. We also provide distances based on *lattice valuations*. An *upper valuation* :math:`\mu : L \to \mathbb{R}` is a monotone function satisfying :math:`\mu(x \wedge y) + \mu(x \vee y) \leq \mu(x) + \mu(y)`. Every lattice has as an upper valuation given by

    :math:`\mu(x) := | L \setminus \{k \in L : k \geq x \}|`

which induces a metric :math:`d(x,y) := 2\mu(x \vee y) - \mu(x) - \mu(y)`. This metric is computed using 

    lat.distance(x,y, metric = 'upper_valuation')

Dually one has an induced *lower valuation* on the opposite lattice which is computed by the distance function for mertic='lower_valuation'. Note that these distances are always distinct, and that we get isometric metric spaces when L is isomorphic to its opposite as a lattice. For concept lattices, one can also give a lower valuation by assigning **positive** real weights to each vertex :math:`w : V \to \mathbb{R}_+`. The lower valuation is then defined on a concept (A,B) by

    :math:`\mu(A,B) := \sum_{v \in A} w(v)`

and the resulting metric is :math:`d( (A,B), (C,D) ) := 2\mu( (A,B) \wedge (C,D) ) - \mu(A,B) - \mu(C,D)`

One may also obtain a dictionary of distances for a desired metric via::

    lat.all_distances(metric)
    

References
^^^^^^^^^^
[1] Leclerc B., "Lattice valuations, medians and majorities", In: Discrete Mathematics, Volume 111, Issues 1–3, 1993, Pages 345-356, https://doi.org/10.1016/0012-365X(93)90169-T

[2] Ganter B., Wille R., Franzke C., "Formal Concept Analysis: Mathematical Foundations", Springer Cham, 1998, https://doi.org/10.1007/978-3-031-63422-2
