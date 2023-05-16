.. _glossary:

=====================
Glossary of HNX terms
=====================


The HNX library centers around the idea of a :term:`hypergraph`.  This glossary provides a few key terms and definitions.


.. glossary::
	:sorted:


	.. // scan hypergraph.py

	Entity and Entity set
		Class in entity.py. 
		HNX stores many of its data structures inside objects of type Entity.  Entities help to insure safe behavior, but their use is primarily technical, not mathematical.

	hypergraph
		The term *hypergraph* can have many different meanings.  In HNX, it means a tuple (Nodes, Edges, Incidence), where Nodes and Edges are sets, and Incidence is a function that assigns a value of True or False to every pair (n,e) in the Cartesian product Nodes x Edges.  We call 
		- Nodes the set of nodes
		- Edges the set of edges
		- Incidence the incidence function
		*Note* Another term for this type of object is a *multihypergraph*.  The ability to work with multihypergraphs efficiently is a distinguishing feature of HNX!

	incidence
		A node n is incident to an edge e in a hypergraph (Nodes, Edges, Incidence) if Incidence(n,e) = True.	
		!!! -- give the line of code that would allow you to evaluate 

	incidence matrix
		A rectangular matrix constructed from a hypergraph (Nodes, Edges, Incidence) where the elements of Nodes index the matrix rows, and the elements of Edges index the matrix columns. Entry (n,e) in the incidence matrix is 1 if n and e are incident, and is 0 otherwise.			

	edge nodes (aka edge elements)
		The nodes (or elements) of an edge e in a hypergraph (Nodes, Edges, Incidence) are the nodes that are incident to e.

	subhypergraph
		A subhypergraph of a hypergraph (Nodes, Edges, Incidence) is a hypergraph (Nodes', Edges', Incidence') such that Nodes' is a subset of Nodes, Edges' is a subset of Edges, and every incident pair (n,e) in (Nodes', Edges', Incidence') is also incident in (Nodes, Edges, Incidence)

	subhypergraph induced by a set of nodes
		An induced subhypergraph of a hypergraph (Nodes, Edges, Incidence) is a subhypergraph (Nodes', Edges', Incidence') where a pair (n,e) is incident if and only if it is incident in (Nodes, Edges, Incidence)

	degree
		Given a hypergraph (Nodes, Edges, Incidence), the degree of a node in Nodes is the number of edges in Edges to which the node is incident.
		See also: :term:`s-degree`		

	dual
		The dual of a hypergraph (Nodes, Edges, Incidence) switches the roles of Nodes and Edges. More precisely, it is the hypergraph (Edges, Nodes, Incidence'), where Incidence' is the function that assigns Incidence(n,e) to each pair (e,n).  The :term:`incidence matrix` of the dual hypergraph is the transpose of the incidence matrix of (Nodes, Edges, Incidence).

	toplex
		A toplex in a hypergraph (Nodes, Edges, Incidence ) is an edge e whose node set isn't properly contained in the node set of any other edge.  That is, if f is another edge and ever node incident to e is also incident to f, then the node sets of e and f are identical.

	simple hypergraph
		A hypergraph for which no edge is completely contained in another.

-------------
S-line graphs
-------------

HNX offers a variety of tool sets for network analysis, including s-line graphs.

	s-adjacency matrix
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, a square matrix where the elements of Nodes index both rows and columns. The matrix can be weighted or unweighted. Entry (i,j) is nonzero if and only if node i and node j are incident to at least s edges in common.  If it is nonzero, then it is equal to the number of shared edges (if weighted) or 1 (if unweighted).

	s-edge-adjacency matrix
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, a square matrix where the elements of Edges index both rows and columns. The matrix can be weighted or unweighted. Entry (i,j) is nonzero if and only if edge i and edge j share to at least s nodes, and is equal to the number of shared nodes (if weighted) or 1 (if unweighted).

	s-auxiliary matrix
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, the submatrix of the :term:`s-edge-adjacency matrix <s-edge-adjacency matrix>` obtained by restricting to rows and columns corresponding to edges of size at least s.

	s-node-walk
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, a sequence of nodes in Nodes such that each successive pair of nodes share at least s edges in Edges.

	s-edge-walk
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, a sequence of edges in Edges such that each successive pair of edges intersects in at least s nodes in Nodes.

	s-walk
		Either an s-node-walk or an s-edge-walk.

	s-connected component, s-node-connected component
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, an s-connected component is a :term:`subhypergraph` induced by a subset of Nodes with the property that there exists an s-walk between every pair of nodes in this subset. An s-connected component is the maximal such subset in the sense that it is not properly contained in any other subset satisfying this property.

	s-edge-connected component
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, an s-edge-connected component is a :term:`subhypergraph` induced by a subset of Edges with the property that there exists an s-edge-walk between every pair of edges in this subset. An s-edge-connected component is the maximal such subset in the sense that it is not properly contained in any other subset satisfying this property.

	s-connected, s-node-connected
		A hypergraph is s-connected if it has one s-connected component.

	s-edge-connected
		A hypergraph is s-edge-connected if it has one s-edge-connected component.

	s-distance
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, the s-distances between two nodes in Nodes is the length of the shortest :term:`s-node-walk` between them. If no s-node-walks between the pair of nodes exists, the s-distance between them is infinite. The s-distance
		between edges is the length of the shortest :term:`s-edge-walk` between them. If no s-edge-walks between the pair of edges exist, then s-distance between them is infinite.

	s-diameter
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, the s-diameter is the maximum s-Distance over all pairs of nodes in Nodes.

	s-degree
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, the s-degree of a node is the number of edges in Edges of size at least s to which node belongs. See also: :term:`degree`

	s-edge
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, an s-edge is any edge of size at least s.

	s-linegraph
		For a hypergraph (Nodes, Edges, Incidence) and positive integer s, an s-linegraph is a graph representing
		the node to node or edge to edge connections according to the *width* s of the connections.
		The node s-linegraph is a graph on the set Nodes. Two nodes in Nodes are incident in the node s-linegraph if they
		share at lease s incident edges in Edges; that is, there are at least s elements of Edges to which they both belong.
		The edge s-linegraph is a graph on the set Edges. Two edges in Edges are incident in the edge s-linegraph if they
		share at least s incident nodes in Nodes; that is, the edges intersect in at least s nodes in Nodes.

	.. Bipartite Condition
	.. 	Condition imposed on instances of the class EntitySet.
	..     *Entities that are elements of the same EntitySet, may not contain each other as elements.* 
	..     The elements and children of an EntitySet generate a specific partition for a bipartite graph. 
	..     The partition is isomorphic to a Hypergraph where the elements correspond to hyperedges and
	..     the children correspond to the nodes. EntitySets are the basic objects used to construct dynamic hypergraphs
	..     in HNX. See methods :py:meth:`classes.hypergraph.Hypergraph.bipartite` and :py:meth:`classes.hypergraph.Hypergraph.from_bipartite`.






