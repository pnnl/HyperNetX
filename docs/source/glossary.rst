.. _glossary:

=====================
Glossary of HNX terms
=====================

.. glossary::
	:sorted:

	Entity
		Class in entity.py. 
		The base class for nodes, edges, and other HNX structures. An entity has a unique id, a set of properties, and a set of other entities belonging to it called its :term:`elements <Entity.elements>` (an entity may not contain itself).
		If an entity A belongs to another entity B then A has membership in B and A is an element of B. For any entity A access a dictionary of its elements (keyed by uid) using ``A.elements`` and a dictionary of its memberships using ``A.memberships``.

	Entity.elements
		Attribute in class Entity. Returns a dictionary of elements of the entity.
		For any entity A, the elements equal the set of entities belonging to A. Use ``A.uidset`` to access the set of uids belonging to the elements of A and ``A.elements`` to access a dictionary of uid,entity key value pairs of elements of A.

	Entity.children
		Attribute in class Entity. Returns a set of uids for the elements of the elements of entity.
		For any entity A, the set of entities which belong to some entity belonging to A.  Use ``A.children`` to access the set of uids belonging to the children of A and ``A.registry`` to access a dictionary of uid,entity key value pairs of the children of A.
		See also :term:`Entity.levelset`.

	Entity.registry
		Attribute in class Entity.
		A dictionary of uid,entity key value pairs of the :term:`children <Entity.children>` of an entity.

	Entity.memberships
		Attribute in class Entity.
		A dictionary of uid,entity key value pairs of entities to which the entity belongs.

	Entity.levelset
		Method in class Entity.
		For any entity A, Level 1 of A is the set of :term:`elements <Entity.elements>` of A.
		The elements of entities in Level 1 of A belong to Level 2 of A. The elements of entities in Level k of A belong to Level k+1 of A.
		The entities in Level 2 of A are called A's children.
		A single entity may occupy multiple Level sets of an entity. An entity may belong to any of its own Level sets except Level 1 as no entity may contain itself as an element.
		Note that if Level n of A is nonempty then Level k of A is nonempty for all k<n.
		Use ``A.levelset(k)`` to access a dictionary of uid,entity key value pairs for the entities in Level k of A.

	Entity.depth
		Method in class Entity.
		The number of non empty :term:`Level sets <Entity.levelset>` belonging to an entity.
		For any entity A, if A.elements is empty then it has depth 0 and no non-empty Levels.
		If A.elements contains only Entities of depth 0 then A has depth 1.
		If A.elements contains only Entities of depth 0 and depth 1 then A has depth 2.
		If A.elements contains an entity of depth n and no Entities of depth more than n then it has depth n+1.

	entityset
		An entity A satisfying the :term:`Bipartite Condition`, the property that the set of entities in Level 1 of A is disjoint from the set of entities in Level 2 of A, i.e. the elements of A are disjoint from the children of A. An entityset is instantiated in the class EntitySet.

	hypergraph
		A pair of entitysets (Nodes,Edges) such that Edges has :term:`depth <Entity.depth>` 2, Nodes have depth 1, and the children of Edges is exactly the set of elements of Nodes. Intuitively, every element of Edges is a (hyper)edge, which is either empty or contains elements of Nodes. Every node in Nodes has :term:`membership <Entity.memberships>` in some edge in Edges. Since a node has :term:`depth <Entity.depth>` 0 it is distinguished by its uid, properties, and memberships. A hypergraph is instantiated in the class Hypergraph.

	subhypergraph
		Given a hypergraph (Nodes,Edges), a subhypergraph is a pair of subsets of (Nodes,Edges).

	degree
		Given a hypergraph (Nodes,Edges), the degree of a node in Nodes is the number of edges in Edges to which the node belongs.
		See also: :term:`s-degree`

	incidence matrix
		A rectangular matrix constructed from a hypergraph (Nodes,Edges) where the elements of Nodes index the matrix rows, and the elements of Edges index the matrix columns. Entry (i,j) in the incidence matrix is 1 if the node corresponding to i in Nodes belongs to the edge corresponding to j in Edges, and is 0 otherwise.

	s-adjacency matrix
		For a hypergraph (Nodes,Edges) and positive integer s, a square matrix where the elements of Nodes index both rows and columns. The matrix can be weighted or unweighted. Entry (i,j) is nonzero if and only if node i and node j belong to at least s shared edges, and is equal to the number of shared edges (if weighted) or 1 (if unweighted).

	s-edge-adjacency matrix
		For a hypergraph (Nodes,Edges) and positive integer s, a square matrix where the elements of Edges index both rows and columns. The matrix can be weighted or unweighted. Entry (i,j) is nonzero if and only if edge i and edge j share to at least s nodes, and is equal to the number of shared nodes (if weighted) or 1 (if unweighted).

	s-auxiliary matrix
		For a hypergraph (Nodes,Edges) and positive integer s, the submatrix of the :term:`s-edge-adjacency matrix <s-edge-adjacency matrix>` obtained by restricting to rows and columns corresponding to edges of size at least s.

	toplex
		For a hypergraph (Nodes,Edges), a toplex is an edge in Edges whose elements (i.e. nodes) do not all belong to any other edge in Edge.

	dual
		For a hypergraph (Nodes,Edges), its dual is the hypergraph constructed by switching the roles of Nodes and Edges. More precisely, if node i belongs to edge j in the hypergraph, then node j belongs to edge i in the dual hypergraph.

	s-node-walk
		For a hypergraph (Nodes,Edges) and positive integer s, a sequence of nodes in Nodes such that each successive pair of nodes share at least s edges in Edges.

	s-edge-walk
		For a hypergraph (Nodes,Edges) and positive integer s, a sequence of edges in Edges such that each successive pair of edges intersects in at least s nodes in Nodes.

	s-walk
		Either an s-node-walk or an s-edge-walk.

	s-connected component, s-node-connected component
		For a hypergraph (Nodes,Edges) and positive integer s, an s-connected component is a :term:`subhypergraph` induced by a subset of Nodes with the property that there exists an s-walk between every pair of nodes in this subset. An s-connected component is the maximal such subset in the sense that it is not properly contained in any other subset satisfying this property.

	s-edge-connected component
		For a hypergraph (Nodes,Edges) and positive integer s, an s-edge-connected component is a :term:`subhypergraph` induced by a subset of Edges with the property that there exists an s-edge-walk between every pair of edges in this subset. An s-edge-connected component is the maximal such subset in the sense that it is not properly contained in any other subset satisfying this property.

	s-connected, s-node-connected
		A hypergraph is s-connected if it has one s-connected component.

	s-edge-connected
		A hypergraph is s-edge-connected if it has one s-edge-connected component.

	s-distance
		For a hypergraph (Nodes,Edges) and positive integer s, the s-distances between two nodes in Nodes is the length of the shortest :term:`s-node-walk` between them. If no s-node-walks between the pair of nodes exists, the s-distance between them is infinite. The s-distance
		between edges is the length of the shortest :term:`s-edge-walk` between them. If no s-edge-walks between the pair of edges exist, then s-distance between them is infinite.

	s-diameter
		For a hypergraph (Nodes,Edges) and positive integer s, the s-diameter is the maximum s-Distance over all pairs of nodes in Nodes.

	s-degree
		For a hypergraph (Nodes, Edges) and positive integer s, the s-degree of a node is the number of edges in Edges of size at least s to which node belongs. See also: :term:`degree`

	s-edge
		For a hypergraph (Nodes, Edges) and positive integer s, an s-edge is any edge of size at least s.

	s-linegraph
		For a hypergraph (Nodes, Edges) and positive integer s, an s-linegraph is a graph representing
		the node to node or edge to edge connections according to the *width* s of the connections.
		The node s-linegraph is a graph on the set Nodes. Two nodes in Nodes are incident in the node s-linegraph if they
		share at lease s incident edges in Edges; that is, there are at least s elements of Edges to which they both belong.
		The edge s-linegraph is a graph on the set Edges. Two edges in Edges are incident in the edge s-linegraph if they
		share at least s incident nodes in Nodes; that is, the edges intersect in at least s nodes in Nodes.

	Bipartite Condition
		Condition imposed on instances of the class EntitySet.
	    *Entities that are elements of the same EntitySet, may not contain each other as elements.* 
	    The elements and children of an EntitySet generate a specific partition for a bipartite graph. 
	    The partition is isomorphic to a Hypergraph where the elements correspond to hyperedges and
	    the children correspond to the nodes. EntitySets are the basic objects used to construct dynamic hypergraphs
	    in HNX. See methods :py:meth:`classes.hypergraph.Hypergraph.bipartite` and :py:meth:`classes.hypergraph.Hypergraph.from_bipartite`.

	simple hypergraph
		A hypergraph for which no edge is completely contained in another.




