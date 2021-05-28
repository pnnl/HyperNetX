.. _nwhy:

====
NWhy
====

Overview
========
		NWhy is a C++-based, scalable, high-performance hypergraph library. It has three dependencies.
		It uses NWGraph library as the building block. NWGraph provides graph data structures, a rich set of adaptors over the graph data structures, various high-performance graph algorithms implementations.
		It relies Intel OneAPI Threading Building Blocks (oneTBB) to provide parallelism.
		To expose the C++ implementations to Python, Pybind11 are adopted to encapsulate NWhy as a python module.
		The goal of the NWhy python APIs is to share a ID space between NWhy and its user for hypergraph processing, instead of copying the sparse matrix of the hypergraph back and forth between NWhy and its user.
		NWhy is developed by Xu Tony Liu. The current version is preliminary and under active development.

Installing NWhy
===============

The NWhy library provides Pybind11_ APIs for analysis of complex data set intepret as hypergraphs.

.. _Pybind11: https://github.com/pybind/pybind11

To install in an Anaconda environment
-------------------------------------

	>>> conda create -n <env name> python=3.9

Then activate the environment
-----------------------------

	>>> conda activate <env name> 

Install Intel Threading Building Blocks(TBB)
--------------------------------------------

To install TBB_:

.. _TBB: https://github.com/oneapi-src/oneTBB

	>>> conda install tbb

If a local TBB has been install, we can specify TBBROOT

    >>> export TBBROOT=/opt/tbb/
	
Install using Pip
-----------------

For installation:

	>>> pip install nwhy

For upgrade:

	>>> pip install nwhy --upgrade

or 

	>>> pip install nwhy -U


Quick test with import
----------------------

For quick test:

	>>> python -c "import nwhy"

If there is no import error, then installation is done.

NWhy APIs
=========

.. _nwhy::
	:sorted:


nwhy module
-----------

	_version
		Attribute in nwhy module.
		Return the version number of nwhy module.


NWHypergraph class
------------------

	NWHypergraph
		Class in nwhy module.
		The base class for hypergraph representation in nwhy. It accepts a directed edge list format of hypergraph, either weighted or unweighted, then construct the NWHypergraph object.

NWHypergraph class attribues
----------------------------

	NWHypergraph.row
		Attribute in class NWHypergraph. 
		Return a Numpy array of IDs, row of sparse matrix of the hypergraph. Note the number of entries in the Numpy lists, row, col and data must be equal. The row stores hyperedges.
	NWHypergraph.col
		Attribute in class NWHypergraph. 
		Return a Numpy array of IDs, columns of sparse matrix of the hypergraph. The col stores vertices.
	NWHypergraph.data
		Attribute in class NWHypergraph. 
		Return a Numpy array of IDs, weights of sparse matrix of the hypergraph.

NWHypergraph class methods
--------------------------

	NWHypergraph.NWHypergraph(x, y)
		Constructor of class NWHypergraph.
		Return a NWHypergraph object. Here the hypergraph is unweighted. X is a Numpy array of hyperedges, and y is a Numpy array of vertices.

	NWHypergraph.NWHypergraph(x, y, data)
		Constructor of class NWHypergraph.
		Return a NWHypergraph object. Here the hypergraph is weighted. X is a Numpy array of hyperedges, y is a Numpy array of vertices, data is a Numpy array of weights associated with the pairs from hyperedges to vertices.

	NWHypergraph.collapse_edges(return_equal_class=False)
		Method in class NWHypergraph.
		Return a dictionary, where the key is a new ID of a hyperedge after collapsing the hyperedges if the hyperedges have the same vertices, and the value is the number of such hyperedges when `return_equal_class=False`, otherwise, the set of such hyperedges when `return_equal_class=True`. Note the weights associated with the pairs from hyperedges to vertices are not collapsed or combined.

	NWHypergraph.collapse_nodes(return_equal_class=False)
		Method in class NWHypergraph.	
		Return a dictionary, where the key is a new ID of a vertex after collapsing the vertices if the vertices share the same hyperedges, and the value is the number of such vertices when `return_equal_class=False`, otherwise, the set of such vertices when `return_equal_class=True`. Note the weights associated with the pairs from hyperedges to vertices are not collapsed or combined.

	NWHypergraph.collapse_nodes_and_edges(return_equal_class=False)
		Method in class NWHypergraph.
		Return a dictionary, where the key is a new ID of a hyperedge after collapsing the hyperedges if the hyperedges share the same vertices, and the value is the number of such hyperedges when `return_equal_class=False`, otherwise, the set of such hyperedges when `return_equal_class=True`. This method is not equivalent to call `NWHypergraph.collapse_nodes()` then `NWHypergraph.collapse_edges()`. Note the weights associated with the pairs from hyperedges to vertices are not collapsed or combined.

	NWHypergraph.edge_size_dist()
		Method in class NWHypergraph.
		Return a list of edge size distribution of the hypergraph.

	NWHypergraph.node_size_dist()
		Method in class NWHypergraph.
		Return a list of vertex size distribution of the hypergraph.

	NWHypergraph.edge_incidence(edge)
		Method in class NWHypergraph.
		Return a list of vertices that are incident to hyperedge `edge`.

	NWHypergraph.node_incidence(node)
		Method in class NWHypergraph.
		Return a list of hyperedges that are incident to vertex `node`.

	NWHypergraph.degree(node, min_size=1, max_size=None)
		Method in class NWHypergraph.
		Return the degree of the vertex `node` in the hypergraph. For the hyperedges `node` incident to, if `min_size` or/and `max_size` are specified, then either/both criteria are used to filter the hyperedges.

	NWHypergraph.size(edge, min_degree=1, max_degree=None)
		Method in class NWHypergraph.
		Return the size of the hyperedge `edge` in the hypergraph. For the vertices `edge` incident to, if `min_degree` or/and `max_degree` are specified, then either/both criteria are used to filter the vertices.

	NWHypergraph.dim(edge)
		Method in class NWHypergraph.
		Return the dimension of the hyperedge `edge` in the hypergraph.

	NWHypergraph.number_of_nodes()
		Method in class NWHypergraph.
		Return the number of vertices in the hypergraph.

	NWHypergraph.number_of_edges()
		Method in class NWHypergraph.
		Return the number of edges in the hypergraph.

	NWHypergraph.singletons()
		Method in class NWHypergraph.
		Return a list of singleton hyperedges in the hypergraph. A singleton hyperedge is incident to only one vertex.
	
	NWHypergraph.toplexes()
		Method in class NWHypergraph.
		Return a list of toplexes in the hypergraph. For a hypergraph (Edges, Nodes), a toplex is a hyperedge in Edges whose elements (i.e. nodes) do not all belong to any other hyperedge in Edge.

	NWHypergraph.s_linegraph(s=1, edges=True)
		Method in class NWHypergraph.
		Return a Slinegraph object. Construct a s-line graph from the hypergraph for a positive integer `s`. In this s-line graph, the vertices are the hyperedges in the original hypergraph if `edges=True`; otherwise, the vertices are the vertices in the original hypergraph. Note this method create s-line graph on the fly, therefore it requires less memory compared with `NWHypergraph.s_linegraphs(l, edges=True)`. It is slower to construct multiple s-line graphs for different `s` compared with `NWHypergraph.s_linegraphs(l, edges=True)`.

	NWHypergraph.s_linegraphs(l, edges=True)
		Method in class NWHypergraph.
		Return a list of Slinegraph objects. For each positive integer in list `l`, construct a Slinegraph object from the hypergraph. In each s-line graph, the vertices are the hyperedges in the original hypergraph if `edges=True`; otherwise, the vertices are the vertices in the original hypergraph. Note this method creates multiple s-line graphs for one run, therefore it is significantly faster compared with `NWHypergraph.s_linegraph(s=1, edges=True)`, but it requires much more memory.


Slinegraph class
----------------

	Slinegraph
		Class in nwhy module.
		The base class for s-line graph representation in nwhy. It store an undirected graph, called an s-line graph of a hypergraph given a positive integer s. Slinegraph can be an 'edge' line graph, where the vertices in Slinegraph are the hyperedges in the original hypergraph; Slinegraph can also be a 'vertex' line graph, where the vertices in Slinegraph are the vertices in the original hypergraph.

Slinegraph class attribues
--------------------------
		
	Slinegraph.row
		Attribute in class Slinegraph. 
		Return a Numpy array of IDs, row of sparse matrix of the s-line graph. Note the number of entries in the Numpy lists, row, col and data must be equal. 
	Slinegraph.col
		Attribute in class Slinegraph. 
		Return a Numpy array of IDs, columns of sparse matrix of the s-line graph.
	Slinegraph.data
		Attribute in class Slinegraph. 
		Return a Numpy array of IDs, weights of sparse matrix of the s-line graph. The weights are not the hyperedge-vertex pair weights. Currently, if Slinegraph is an edge line graph, the weights are the number of overlapping vertices between two hyperedges in the original hypergraph. If the Slinegraph is a vertex line graph, the weights are the number of overlapping hyperedges between two vertices in the original hypergraph.
	Slinegraph.s
		Attribute in class Slinegraph. 
		Return s value of the s-line graph.

Slinegraph class methods
------------------------

	Slinegraph.Slinegraph(g, s=1, edges=True)
		Constructor of class Slinegraph.
		Return a new Slinegraph object. Given a positive integer `s`, construct a s-line graph from the hypergraph `g`. The vertices in the s-line graph are the hyperedges in `g` if `edges=True`, otherwise, the vertices in the s-line graph are the vertices in `g`.

	Slinegraph.Slinegraph(x, y, data, s=1, edges=True)
		Constructor of class Slinegraph.
		Return a new Slinegraph object. Given an edge list format of a s-line graph stored in three Numpy arrays, construct a s-line graph from the edge list. A positive integer `s` and a boolean `edges` are required to indicate the properties of the s-line graph.

	Slinegraph.get_singletons()
		Method in class Slinegraph.
		Return a list of singletons in the s-line graph.

	Slinegraph.s_connected_components()
		Method in class Slinegraph.
		Return a list of sets, where each set contains the vertices sharing the same component.

	Slinegraph.is_s_connected()
		Method in class Slinegraph.
		Return True or False. Check whether s-line graph is connected.

	Slinegraph.s_distance(src, dest)
		Method in class Slinegraph.
		Return the distance from `src` to `dest`. Return -1 if it is unreachable from `src` to `dest`.

	Slinegraph.s_diameter(src, dest)
		Method in class Slinegraph.
		Return the diameter of the s-line graph. Return 0 if every vertex is a singleton.

	Slinegraph.s_path(src, dest)
		Method in class Slinegraph.
		Return a list of vertices. The vertices are the vertices on the shortest path from `src` to `dest` in the s-line graph. The list will be empty if it is unreachable from `src` to `dest`.

	Slinegraph.s_betweenness_centrality(normalized=True)
		Method in class Slinegraph.
		Return a list of betweenness centrality score of every vertices in the s-line graph. The betweenness centrality score will be normalized by 2/((n-1)(n-2)) if `normalized=True` where n the number of vertices in s-line graph.  Betweenness centrality of a vertex `v` is the sum of the fraction of all-pairs shortest paths that pass through `v`: 

		.. math::

			c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

	Slinegraph.s_closeness_centrality(v=None)
		Method in class Slinegraph.
		Return a list of closeness centrality scores of every vertices in the s-line graph. If `v` is specified, then the list returned contains only `v`'s score. Closeness centrality of a vertex `v` is the reciprocal of the average shortest path distance to `v` over all `n-1` reachable nodes:

    	.. math::

        	C(v) = \frac{n - 1}{\sum_{v=1}^{n-1} d(u, v)},


	Slinegraph.s_harmonic_closeness_centrality(v=None)
		Method in class Slinegraph.
		Return a list of harmonic closeness centrality scores of every vertices in the s-line graph. If `v` is specified, then the list returned contains only `v`'s score. Harmonic centrality of a vertex `v` is the sum of the reciprocal of the shortest path distances from all other nodes to `v`:
	
		.. math::
	
			C(v) = \sum_{v \neq u} \frac{1}{d(v, u)}

	Slinegraph.s_eccentricity(v=None)
		Method in class Slinegraph.
		Return a list of eccentricity of every vertices in the s-line graph. If `v` is specified, then the list returned contains only eccentricity of `v`.
			
	Slinegraph.s_neighbors(v)
		Method in class Slinegraph.
		Return a list of neighboring vertices of `v` in the s-line graph.

	Slinegraph.s_degree(v)
		Method in class Slinegraph.
		Return the degree of vertex `v` in the s-line graph.

