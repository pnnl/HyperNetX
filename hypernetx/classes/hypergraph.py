# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import warnings
from hypernetx.classes.entity import Entity, EntitySet
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from scipy import sparse
from hypernetx.exception import HyperNetXError



class Hypergraph():
	"""
	Hypergraph H = (V,E) references a pair of disjoint sets: 
	V = nodes (vertices) and E = (hyper)edges E. 

	The objects in V and E must be distinguishable entities,
	allowing for multi-edge graphs and inseperable nodes. 
	For example: Let V = {1,2,3} and E = {e1,e2,e3},
	where e1 = {1,2}, e2 = {1,2}, and e3 = {1,2,3}. 
	The edges e1 and e2 contain the same set of nodes and yet
	are distinct and must be distinguishable within H.

	To keep track of the objects in a hypergraph each node and edge is 
	instantiated as an Entity and given an identifier, uid. Since 
	hypergraphs can be quite large, only these identifiers will be used 
	for computation intensive methods, this means the user must take care
	to keep a one to one correspondence between their set of uids and 
	the objects in their hypergraph. See `Honor System`_ 

	We will create hypergraphs in multiple ways:

	1. As an empty instance: ::

		>>> H = hnx.Hypergraph()
		>>> H.nodes, H.edges
		({}, {})

	2. From a dictionary of iterables (elements of iterables must be of type hypernetx.Entity or hashable) ::

		>>> H = Hypergraph({'a':[1,2,3],'b':[4,5,6]})
		>>> H.nodes, H.edges
		(EntitySet(_:Nodes,[1, 2, 3, 4, 5, 6],{}), EntitySet(_:Edges,['b', 'a'],{}))

	3. From an iterable of iterables: (elements of iterables must be of type hypernetx.Entity or hashable) ::

		>>> H = Hypergraph([{'a','b'},{'b','c'},{'a','c','d'}])
		>>> H.nodes, H.edges
		(EntitySet(_:Nodes,['d', 'b', 'c', 'a'],{}),
 		 EntitySet(_:Edges,['_1', '_2', '_0'],{}))

	4. From a hypernetx.EntitySet ::

		>>> a = Entity('a',{1,2}); b = Entity('b',{2,3})
		>>> E = EntitySet('sample',elements=[a,b])
		>>> H = Hypergraph(E)
		>>> H.nodes, H.edges
		(EntitySet(_:Nodes,[1, 2, 3],{}), EntitySet(_:Edges,['b', 'a'],{}))

	5. From a networkx bipartite graph using :code:`from_bipartite()`:

		>>> import networkx as nx
		>>> B = nx.Graph()
		>>> B.add_nodes_from([1, 2, 3, 4], bipartite=0)
		>>> B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
		>>> B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
		>>> H = Hypergraph.from_bipartite(B)
		>>> H.nodes, H.edges
		(EntitySet(_:Nodes,[1, 2, 3, 4],{}), EntitySet(_:Edges,['b', 'c', 'a'],{}))

	Parameters
	----------
	setsystem : EntitySet or dictionary or iterable of hashables or Entities, optional, default: None
		If not an EntitySet then setsystem must be acceptable as elements to an EntitySet

	name : hashable, optional, default: None

	"""


	def __init__(self,setsystem=None, name='_'): 

		self.name = name

		##### Check setsystem type and change into an EntitySet before constructing hypergraph:

		if not setsystem:
			setsystem = EntitySet('_',elements=[])

		elif isinstance(setsystem,dict):
			### Must be a dictionary with values equal to iterables of Entities and hashables.
			### Keys will be uids for new edges and values of the dictionary will generate the nodes.
			setsystem = EntitySet('_',setsystem)

		### If no ids are given, return default ids indexed by position in iterator
		### This should be an iterable of sets
		elif not isinstance(setsystem, EntitySet):
			labels = [self.name+str(x) for x in range(len(setsystem))]
			setsystem = EntitySet('_',dict(zip(labels, setsystem)))

		_reg = setsystem.registry
		_nodes = {k: Entity(k,**_reg[k].properties) for k in _reg }
		_elements = {j: {k : _nodes[k] for k in setsystem[j]} 
						for j in setsystem}
		_edges = {j: Entity(j, 
					elements = _elements[j].values(), 
					**setsystem[j].properties) for j in setsystem}

		self._edges = EntitySet(f'{self.name}:Edges', 
						elements = _edges.values(), **setsystem.properties)
		self._nodes = EntitySet(f'{self.name}:Nodes', 
						elements = _nodes.values())


	@property
	def edges(self):
		"""
		Dictionary of EntitySet of (hyper)edges
		"""
		return self._edges

	@property
	def nodes(self):
		"""
		Dictionary of EntitySet of nodes
		"""
		return self._nodes


	@property
	def incidence_dict(self):
		"""
		Dictionary keyed by edge uids with values the uids of nodes in each edge
		"""
		return self._edges.incidence_dict

	@property
	def shape(self):
		"""
		Tuple giving (number of nodes, number of edgess)
		"""

		return (len(self._nodes),len(self._edges))

	def __str__(self):
		"""
		String representation of hypergraph
		"""
		return f'Hypergraph({self.edges.elements},name={self.name})'

	def __repr__(self):
		"""
		String representation of hypergraph
		"""
		return f'Hypergraph({self.edges.elements},name={self.name})'

	def __len__(self):
		"""
		Number of nodes
		"""
		return len(self._nodes)

	def __eq__(self,other):
		"""
		Determine if two hypergraphs are equal.
		
		Parameters
		----------
		other : Hypergraph

		Returns
		-------
		boolean : boolean 
		
		Notes
		-----
		Two Hypergraphs are equal if they have the same name and
		are generated by the same EntitySet(=self.edges). If one of them does not
		have a name only the EntitySets are checked.

		"""
		assert isinstance(other,Hypergraph)
		if self.name and other.name and self.name is not other.name:
			return False
		else:
			return self._edges.incidence_dict == other._edges.incidence_dict

	def __iter__(self):
		"""
		Iterate over the nodes of the hypergraph
		"""
		return iter(self.nodes)

	def __contains__(self,item):
		"""
		Returns boolean indicating if item is in self.nodes
		
		Parameters
		----------
		item : hashable or Entity

		"""
		if isinstance(item,Entity):
			return item.uid in self.nodes
		else:
			return item in self.nodes

	def __getitem__(self,node):
		"""
		Return the neighbors of node

		Parameters
		----------
		node : Entity or hashable
			If hashable, then must be uid of node in hypergraph
		
		Returns
		-------
		neighbors(node) : iterator

		"""
		return self.neighbors(node)

	def s_degree(self,node,s=1,edges=None):
		"""
		Return the degree of a node in H when restricted to edges

		Parameters
		----------
		node : Entity or hashable
			If hashable, then must be uid of node in hypergraph

		s : positive integer, optional, default: 1

		edges : iterable of edge.uids, optional, default: None

		Returns
		------- 
		s_degree : int
			The degree of a node in the subgraph induced by edges
			if edges = None return the s-degree of the node

		Note
		----
		The :term:`s-degree` of a node is the number of edges of size
		at least s that contain the node. 

		"""
		return self.degree(node,s,edges)

	def degree(self,node,s=1,edges=None):
		"""
		Return the degree of a node in H

		Parameters
		----------
		node : node.uid

		s : positive integer, optional

		edges : iterable of edge.uids, optional


		Returns
		------- 
			The degree of a node in the subgraph induced by edges
			if edges = None return the s-degree of the node

		Notes
		-----
			The s-degree of a node is the number of edges of size
			at least s that contain the node.

		"""
		memberships = set(self.nodes[node].memberships)
		if edges:
			memberships = memberships.intersection(edges)
		if s>1:
			return len(set(e for e in memberships if len(self.edges[e])>=s))
		else:
			return len(memberships)

	def number_of_nodes(self,nodeset=None):
		"""
		The number of nodes in nodeset belonging to hypergraph.
		
		Parameters
		----------
		nodeset : an interable of Entities, optional, default: None
			If None, then return the number of nodes in hypergraph.

		Returns
		-------
		number_of_nodes : int

		"""
		if nodeset:
			return len([n for n in self.nodes if n in nodeset])
		else:
			return len(self.nodes)

	def number_of_edges(self,edgeset=None):
		"""
		The number of edges in edgeset belonging to hypergraph.
		
		Parameters
		----------
		edgeset : an interable of Entities, optional, default: None
			If None, then return the number of edges in hypergraph.

		Returns
		-------
		number_of_edges : int
		"""
		if edgeset:
			return len([e for e in self.edges if e in edgeset])
		else:
			return len(self.edges)

	def order(self):
		"""
		The number of nodes in hypergraph.

		Returns
		-------
		order : int
		"""
		return len(self.nodes)

	def size(self,edge,nodes=None):
		"""
		The number of nodes in nodes that belong to edge.
		If nodes=None, returns the number of nodes in edge.

		Parameters
		----------
		edge : hashable
			The uid of an edge in the hypergraph

		nodes : iterable, optional, default: hypergraph.nodes.uidset
			An iterable of uids for nodes in hypergraph.

		Returns
		-------
		size : int

		"""
		if nodes:
			return len(self.edges[edge].uidset.intersection(nodes))
		else:
			return len(self.edges[edge])

	def dim(self,edge):
		"""
		Same as size(edge)-1.
		"""

		if edge in self.edges:
			return self.edges[edge].size() - 1
		else:
			return None

	def neighbors(self, node, s=2):
		"""
		The nodes in hypergraph which share an :term:`s-edge` with node.

		Parameters
		----------
		node : hashable
			uid for a node in hypergraph

		s : int, optional, default : 2
			The desired size of the edge connecting node to its neighbors

		Returns
		-------
		neighbors : iterator

		"""
		memberships = set(self.nodes[node].memberships).intersection(self.edges.uidset)
		edges = [e for e in memberships if len(self.edges[e]) >=s]
		neigh = set()
		for e in edges:
			neigh.update(self.edges[e].uidset)
		neigh.discard(node)
		return iter(neigh)


	def remove_node(self,node):
		"""
		Removes node from edges and deletes reference in hypergraph nodes

		Parameters
		----------
		node : hashable or Entity
			a node in hypergraph

		Returns
		-------
		hypergraph : Hypergraph

		"""
		if not node in self._nodes:
			return self
		else:
			if not isinstance(node,Entity):
				node = self._nodes[node]
			for edge in node.memberships:
				self._edges[edge].remove(node)
			self._nodes.remove(node)
		return self

	def remove_nodes(self,node_set):
		"""
		Removes nodes from edges and deletes references in hypergraph nodes

		Parameters
		----------
		node_set : an iterable of hashables or Entities
			Nodes in hypergraph

		Returns
		-------
		hypergraph : Hypergraph

		"""
		for node in node_set:
			self.remove_node(node)
		return self

	def _add_nodes_from(self,nodes):
		"""
		Private helper method instantiates new nodes when edges added to hypergraph. 

		Parameters
		----------
		nodes : iterable of hashables or Entities

		"""
		for node in nodes:
			if node in self._edges:
				raise HyperNetxError("Node already an edge.")
			elif node in self._nodes and isinstance(node,Entity):
				self._nodes[node].__dict__.update(node.properties)  
			elif node not in self._nodes:
				if isinstance(node,Entity):
					self._nodes.add(Entity(node.uid, **node.properties)) 
				else:
					self._nodes.add(Entity(node)) 

	def add_edge(self, edge, **kwargs):
		"""
		Adds a single edge to hypergraph.

		Parameters
		----------
		edge : hashable or Entity
			If hashable the edge returned will be empty.
		kwargs : keyword arguments, optional
            Edge data can be assigned using keyword arguments.

		Returns
		-------
		hypergraph : Hypergraph

		Notes
		-----
		When adding an edge to a hypergraph children must be removed 
		so that nodes do not have elements.
		Each node (element of edge) must be instantiated as a node, 
		making sure its uid isn't already present in the self.
		If an added edge contains nodes that cannot be added to hypergraph
		then an error will be thrown.

		"""
		if edge in self._edges:
			warnings.warn("Cannot add edge. Edge already in hypergraph")  
		elif edge in self._nodes:
			warnings.warn("Cannot add edge. Edge is already a Node")
		elif isinstance(edge,Entity):
			if len(kwargs) > 0:
				warnings.warn(
					("Additional parameters provided "+
					"will be ignored as edge is an Entity.")
				)
			if len(edge) > 0:
				self._add_nodes_from(edge.elements.values())
				self._edges.add(Entity(edge.uid,
					elements=[self._nodes[k] for k in edge], **edge.properties))
				for n in edge.elements:
					self._nodes[n].memberships[edge.uid] = self._edges[edge.uid]
			else:
				self._edges.add(Entity(edge.uid, **edge.properties))		
		else:
			self._add_nodes_from(edge)
			self._edges.add(Entity(edge, **kwargs))  ### this generates an empty edge
			for node in edge:
				self.add_node_to_edge(node, edge)
		return self

	def add_edges_from(self, edge_set, **kwargs):
		"""
		Add edges to hypergraph.

		Parameters
		----------
		edge_set : iterable of hashables or Entities
			For hashables the edges returned will be empty.
		kwargs : keyword arguments, optional
            Edge data (the same for all edges in edge_set
			can be assigned using keyword arguments.

		Returns
		-------
		hypergraph : Hypergraph

		"""
		for edge in edge_set:
			self.add_edge(edge, **kwargs)
		return self


	def add_node_to_edge(self,node,edge):
		"""
		Adds node to an edge in hypergraph edges

		Parameters
		----------
		node: hashable or Entity 
			If Entity, only uid and properties will be used.
			If uid is already in nodes then the known node will 
			be used

		edge: uid of edge or edge, must belong to self.edges

		Returns
		-------
		hypergraph : Hypergraph

		"""
		if edge in self._edges:
			if not isinstance(edge,Entity):
				edge = self._edges[edge]
			if node in self._nodes:
				if not isinstance(node,Entity):
					node = self._nodes[node]
				self._edges[edge].add(node)
			else:
				if not isinstance(node,Entity):
					node = Entity(node)
				else:
					node = Entity(node.uid, **node.properties)
				self._nodes.add(node)
				self._edges[edge].add(node)
		return self



	def remove_edge(self,edge):
		"""
		Removes a single edge from hypergraph.

		Parameters
		----------
		edge : hashable or Entity

		Returns
		-------
		hypergraph : Hypergraph
		
		Notes
		-----

		Deletes reference to edge from all of its nodes. 
		If any of its nodes do not belong to any other edges
		the node is dropped from self.

		"""
		if edge in self._edges:
			if not isinstance(edge,Entity):
				edge = self._edges[edge]
			for node in edge.uidset:
				edge.remove(node)
				if len(self._nodes[node]._memberships) == 1:
					self._nodes.remove(node)
			self._edges.remove(edge)
		return self

	def remove_edges(self,edge_set):
		"""
		Removes edges from hypergraph.

		Parameters
		----------
		edge_set : iterable of hashables or Entities

		Returns
		-------
		hypergraph : Hypergraph

		"""
		for edge in edge_set:
			self.remove_edge(edge)
		return self

	def incidence_matrix(self,sparse=True,index=False):
		"""
        An incidence matrix for the hypergraph indexed by nodes x edges.
        
        Parameters
        ----------
        sparse : boolean, optional, default: True

        index : boolean, optional, default False
            If True return will include a dictionary of node uid : row number
            and edge uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying rows with nodes

        column dictionary : dict
            Dictionary identifying columns with edges

        """
		return self.edges.incidence_matrix(sparse,index)


	def __incidence_to_adjacency(M, s=1, weighted=True):
		"""	
		Helper method to obtain adjacency matrix from incidence matrix.
		
		Parameters
		----------
		M : scipy.sparse.csr.csr_matrix

		s : int, optional, default: 1

		weighted : boolean, optional, default: True
		
		weight: list, optional. A list of weights.

		Returns
		-------
		a matrix : scipy.sparse.csr.csr_matrix

		"""

		A = M.dot(M.transpose())
		A.setdiag(0.0)
		if s > 1:
			A = A.multiply(A >= s)
		if not weighted:
			A = (A > 0)*1
		return A

	def __weighted_incidence_matrix(self, weight, index):
	
		"""
		Helper method to calculate the weighted incidence matrix M * sqrt(W)
		
		Parameters
		----------
		index: boolean
			if True, will return a rowdict of row to node uid	

		weight: str, the name of an element in the edge dict to 
			use as weight in the adjacency matrix calculation
		
		Returns
		----------
		adjacency_matrix : scipy.sparse.csr.csr_matrix
		"""
	
		# this will be a bottleneck in big graphs
		weight_vec = [np.sqrt(self.edges.elements[ii].__dict__[weight]) for ii in self.edges.elements]
		return self.incidence_matrix(index=index).dot(sparse.csr_matrix(np.diag(weight_vec)))

	def adjacency_matrix(self, index=False, s=1, weighted=True, weight=None):
		"""
		The sparse weighted :term:`s-adjacency matrix`
		
		Parameters
		----------
		s : int, optional, default: 1

		index: boolean, optional, default: False
			if True, will return a rowdict of row to node uid

		weighted: boolean, optional, default: True
		
		weight: str, optional, the name of an element in the edge dict to 
			use as weight in the adjacency matrix calculation

		Returns
		-------
		adjacency_matrix : scipy.sparse.csr.csr_matrix

		row dictionary : dict

		Notes
		-----
		If weighted is True each off diagonal cell will equal the number 
		of edges shared by the nodes indexing the row and column if that number is 
		greater than s, otherwise the cell will equal 0. If weighted is False, the off 
		diagonal cell will equal 1 if the nodes indexed by the row and column share at
		least s edges and 0 otherwise.

		"""
		# Note, the name of the keyword "weighted" is unfortunate as it invites confusion
		# with the new keyword "weight," which I've used because that is the convention
		# in networkx. Happy for one or other to be renamed. :)		
		
		if weight is not None:
			M = self.__weighted_incidence_matrix(weight, index)
		else:
			M = self.incidence_matrix(index=index)
			
		if index:
			return Hypergraph.__incidence_to_adjacency(M[0],s=s,weighted=weighted), M[1]
		else:
			return Hypergraph.__incidence_to_adjacency(M,s=s,weighted=weighted)

	def edge_adjacency_matrix(self, index=False, s=1, weighted=True, weight=None):
		"""
		The sparse weighted :term:`s-adjacency matrix` for the dual hypergraph.

		Parameters
		----------
		s : int, optional, default: 1

		index: boolean, optional, default: False
			if True, will return a coldict of column to edge uid

		weighted: boolean, optional, default: True

		weight: str, optional, the name of an element in the edge dict to 
			use as weight in the adjacency matrix calculation (Note, not implemented)

		Returns
		-------
		edge_adjacency_matrix : scipy.sparse.csr.csr_matrix

		column dictionary : dict

		Notes
		-----
		This is also the adjacency matrix for the line graph.
		Two edges are s-adjacent if they share at least s nodes.
		If index=True, returns a dictionary column_index:edge_uid

		"""		
		if weight is not None:
			M = self.__weighted_incidence_matrix(weight, index)
		else:
			M = self.incidence_matrix(index=index)		
		
		if index:
			return Hypergraph.__incidence_to_adjacency(M[0].transpose(),s=s,weighted=weighted), M[2]
		else:
			return Hypergraph.__incidence_to_adjacency(M.transpose(),s=s,weighted=weighted)


	def auxiliary_matrix(self, s=1):
		"""
		The sparse unweighted :term:`s-auxiliary matrix` for hypergraph

		Parameters
		----------
		s : int

		Returns
		-------
		auxiliary_matrix : scipy.sparse.csr.csr_matrix

		Notes
		-----
		Creates subgraph by restricting to edges of cardinality at least s.
		Returns the unweighted s-edge adjacency matrix for the subgraph.

		"""
		edges = [e for e in self.edges if self.edges[e].size() >=s]
		H = self.restrict_to_edges(edges)
		return H.edge_adjacency_matrix(s=s, weighted=False)

	def bipartite(self,node_label=0,edge_label=1):
		"""
		Constructs the networkX bipartite graph associated to hypergraph.

		Parameters
		----------
		node_label : hashable

		edge_label : hashable

		Returns
		-------
		bipartite : nx.Graph() 

		Notes
		-----
		Creates a bipartite networkx graph from hypergraph.
		The nodes and (hyper)edges of hypergraph become the nodes of bipartite graph.
		For every (hyper)edge e in the hypergraph and node n in e there is an edge (n,e)
		in the graph.
		The labels indicate the bipartite partition to use when defining the graph.
		"""
		B = nx.Graph()
		E = self.edges
		V = self.nodes
		B.add_nodes_from(E,bipartite=edge_label)
		B.add_nodes_from(V,bipartite=node_label)
		B.add_edges_from([(v,e) for e in E for v in V if v in E[e]])
		return B

	def dual(self, name=None):
		"""
		Constructs a new hypergraph with roles of edges and nodes of hypergraph reversed.

		Parameters
		----------
		name : hashable

		Returns
		-------
		dual : hypergraph
		"""
		from collections import defaultdict
		E = defaultdict(list)
		for k,v in self.edges.incidence_dict.items():
			for n in v:
				E[n].append(k)
		return Hypergraph(E,name=name)

	def collapse_edges(self, name=None, use_reps=False, return_counts=True):
		"""
		Constructs a new hypergraph gotten by identifying edges containing the same nodes

		Parameters
		----------
		name : hashable, optional, default: None

		use_reps : boolean, optional, default: False
			Choose a single edge from the collapsed edges as uid for the new edge, otherwise uses
			a frozen set of the uids of edges in the equivalence class

		return_counts: boolean, optional, default: True
			if use_reps is True the new edges are keyed by a tuple of the rep and the count

		Returns
		-------
		new hypergraph : Hypergraph

		Notes
		-----
		Two edges are identified if their respective elements are the same. 
		Using this as an equivalence relation, the uids of the edges are partitioned into
		equivalence classes. A frozenset of equivalent edges serves as uid
		for each edge entity.

		If use_reps=True the frozen sets will be replaced with a representative
		from the equivalence classes.

		Example
		-------

			>>> h = Hypergraph(EntitySet('example',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])]))
			>>> h.incidence_dict
			{'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
			>>> h.collapse_edges().incidence_dict
			{frozenset({'E1', 'E2'}): {'a', 'b'}}
			>>> h.collapse_edges(use_reps=True).incidence_dict
			{('E1', 2): {'a', 'b'}}

		"""
		return Hypergraph(self.edges.collapse_identical_elements('_',use_reps=use_reps, return_counts=return_counts), name)

	def collapse_nodes(self, name=None, use_reps=False, return_counts=True):
		"""
		Constructs a new hypergraph gotten by identifying nodes contained by the same edges

		Parameters
		----------        
		use_reps : boolean, optional, default: False
			Choose a single element from the collapsed nodes as uid for the new node, otherwise uses 
			a frozen set of the uids of nodes in the equivalence class

		return_counts: if use_reps is True the new nodes have uids given by a tuple of the rep
			and the count

		Returns
		-------
		new hypergraph : Hypergraph

		Notes
		-----
		Two nodes are identified if their respective memberships are the same. 
		Using this as an equivalence relation, the uids of the nodes are partitioned into
		equivalence classes. A frozenset of equivalent nodes serves as uid
		for each node entity.

		Example
		-------

			>>> h = Hypergraph(EntitySet('example',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])]))
			>>> h.incidence_dict
			{'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
			>>> h.collapse_nodes().incidence_dict
			{'E1': {frozenset({'a', 'b'})}, 'E2': {frozenset({'a', 'b'})}}
			>>> h.collapse_nodes(use_reps=True).incidence_dict
			{'E1': {('a', 2)}, 'E2': {('a', 2)}}

		"""

		return Hypergraph(self.dual().edges.collapse_identical_elements('_',use_reps=use_reps,return_counts=return_counts),name).dual()

	def collapse_nodes_and_edges(self,name=None, use_reps=False, return_counts=True):
		"""
		Returns a new hypergraph by collapsing nodes and edges.

		Parameters
		----------

		use_reps: boolean, optional, default: False
			Choose a single element from the collapsed elements as a representative

		return_counts: boolean, optional, default: True
			if use_reps is True the new elements are keyed by a tuple of the rep
			and the count

		Returns
		-------
		new hypergraph : Hypergraph

		Notes
		-----
		Collapses the Nodes and Edges EntitySets. Two nodes(edges) are duplicates
		if their respective memberships(elements) are the same. Using this as an
		equivalence relation, the uids of the nodes(edges) are partitioned into
		equivalence classes. A frozenset of equivalent nodes(edges) serves as unique id
		for each node(edge) entity.

		Example
		-------

			>>> h = Hypergraph(EntitySet('example',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])]))
			>>> h.incidence_dict
			{'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
			>>> h.collapse_nodes_and_edges().incidence_dict
			{frozenset({'E1', 'E2'}): {frozenset({'a', 'b'})}}
			>>> h.collapse_nodes_and_edges(use_reps=True).incidence_dict
			{('E1', 2): {('a', 2)}}

		"""

		temp = self.collapse_nodes(name=name,use_reps=use_reps,return_counts=return_counts)
		return temp.collapse_edges(name=name,use_reps=use_reps,return_counts=return_counts)
	
	def restrict_to_edges(self,edgeset,name=None):
		"""
		Constructs a hypergraph using a subset of the edges in hypergraph

		Parameters
		----------
		edgeset: iterable of hashables or Entities
			A subset of elements of the hypergraph edges

		name: str, optional, default: None

		Returns
		-------
		new hypergraph : Hypergraph
		"""
		name = name or self.name
		return Hypergraph({e:self.edges[e] for e in edgeset},name)


	def restrict_to_nodes(self,nodeset,name=None):
		"""
		Constructs a new hypergraph by restricting the edges in the hypergraph to
		the nodes referenced by nodeset.

		Parameters
		----------
		nodeset: iterable of hashables
			References a subset of elements of self.nodes

		name: string, optional, default: None

		Returns
		-------
		new hypergraph : Hypergraph
		"""
		memberships = set()
		for node in nodeset:
			if node in self.nodes:
				memberships.update(set(self.nodes[node].memberships))
		newedgeset = dict()
		for e in memberships:
			if e in self.edges:
				temp = self.edges[e].uidset.intersection(nodeset)
				if temp:
					newedgeset[e] = Entity(e,temp,**self.edges[e].properties)
		return Hypergraph(newedgeset,name) 

	def toplexes(self,name=None,collapse=False,use_reps=False,return_counts=True): 
		"""
		Returns a :term:`simple hypergraph` corresponding to self.

		Warning
		------- 
		Collapsing a hypergraph can take a long time. It may be preferable to collapse the graph first and
		pickle it, then apply the toplexes method separately.

		Parameters
		----------
		name: str, optional, default: None

		collapse: boolean, optional, default: False
			Should the hypergraph be collapsed? This would preserve a link between duplicate maximal sets.
			If False then only one of these sets will be used and uniqueness will be up to sets of equal size.

		use_reps: boolean, optional, default: False
			If collapse=True then each toplex will be named by a representative of the set of
			equivalent edges, default is False (see collapse_edges).

		return_counts: boolean, optional, default: True
			If collapse=True then each toplex will be named by a tuple of the representative
			of the set of equivalent edges and their count

		"""
		if collapse:
			if len(self.edges) > 20:  ### TODO: Determine how big is too big.
				warnings.warn('Collapsing a hypergraph can take a long time. It may be preferable to collapse the graph first and pickle it then apply the toplex method separately.')
			temp = self.collapse_edges(use_reps=use_reps,return_counts=return_counts)
		else:
			temp = self
		thdict = dict()
		for e in temp.edges:
			thdict[e] = temp.edges[e].uidset
		tops = dict()
		for e in temp.edges: 
			flag = True
			old_tops = dict(tops)
			for top in old_tops:
				if thdict[e].issubset(thdict[top]):
					flag = False
					break
				elif set(thdict[top]).issubset(thdict[e]):
					del tops[top]
			if flag:
				tops.update({e : thdict[e]})
		return Hypergraph(tops,name)

	def is_connected(self,s=1,edges=False):
		"""
		Determines if hypergraph is :term:`s-connected <s-connected, s-node-connected>`.

		Parameters
		----------
		s: int, optional, default: 1

		edges: boolean, optional, default: False
			If True, will determine if s-edge-connected.
			For s=1 s-edge-connected is the same as s-connected. 

		Returns
		-------
		is_connected : boolean

		Notes
		-----

		A hypergraph is s node connected if for any two nodes v0,vn
		there exists a sequence of nodes v0,v1,v2,...,v(n-1),vn
		such that every consecutive pair of nodes v(i),v(i+1) 
		share at least s edges.

		A hypergraph is s edge connected if for any two edges e0,en
		there exists a sequence of edges e0,e1,e2,...,e(n-1),en
		such that every consecutive pair of edges e(i),e(i+1)
		share at least s nodes.

		"""
		if edges:
			A = self.edge_adjacency_matrix(s=s)
		else:
			A = self.adjacency_matrix(s=s)
		G = nx.from_scipy_sparse_matrix(A)
		return nx.is_connected(G)

	def singletons(self):
		"""
		Returns a list of singleton edges. A singleton edge is an edge of 
		size 1 with a node of degree 1.

		Returns
		-------
		singletons : list
			A list of edge uids.
		"""
		M,r_,cdict = self.incidence_matrix(index=True)
		idx = np.argmax(M.shape) ## which axis has fewest members? if 1 then columns
		cols = M.sum(idx) ## we add down the row index if there are fewer columns
		edges_to_discard = list()
		for c in range(cols.shape[(idx+1)%2]): ## index along opposite axis
			if cols[idx*c,c*((idx+1)%2)] == 1:
				# then see if the singleton entry in that column is also singleton in its row
				# find the entry
				if idx == 0:
					r = np.argmax(M.getcol(c))
					# and get its sum
					s = np.sum(M.getrow(r))
					# if this is also 1 then the entry in r,c represents a singleton
					# so we want to change that entry to 0 and remove the row.
					# this means we want to remove the edge corresponding to c
					if s == 1:
						edges_to_discard.append(cdict[c])
				else: #switch the role of r and c
					r = np.argmax(M.getrow(c))
					s = np.sum(M.getcol(r))
					if s == 1:
						edges_to_discard.append(cdict[r])
		return edges_to_discard	 

	def remove_singletons(self, name=None):
		"""
		Constructs clone of hypergraph with singleton edges removed.
		
		Parameters
		----------
		name: str, optional, default: None
		
		Returns
		-------
		new hypergraph : Hypergraph

		"""
		singles = self.singletons()
		edgeset = [e for e in self._edges if e not in singles]
		return Hypergraph({e:self.edges[e] for e in edgeset},name)

	def s_connected_components(self,s=1,edges=True):   
		"""
		Returns a generator for the :term:`s-edge-connected components <s-edge-connected component>` 
		or the :term:`s-node-connected components <s-connected component, s-node-connected component>` 
		of the hypergraph. 

		Parameters
		----------
		s: int, optional, default: 1 

		edges: boolean, optional, default: True
			If True will return edge components, if False will return node components

		Returns
		-------
		s_connected_components: iterator 
			Iterator returns sets of uids of the edges (or nodes) in the s-edge(node) components of hypergraph. 

		Notes
		-----
		If edges=True, this method returns the s-edge-connected components as 
		lists of lists of edge uids.
		An s-edge-component has the property that for any two edges e1 and e2
		there is a sequence of edges starting with e1 and ending with e2
		such that pairwise adjacent edges in the sequence intersect in at least 
		s nodes. If s=1 these are the path components of the hypergraph.

		If edges=False this method returns s-node-connected components.
		A list of sets of uids of the nodes which are s-walk connected. 
		Two nodes v1 and v2 are s-walk-connected if there is a 
		sequence of nodes starting with v1 and ending with v2 such that pairwise 
		adjacent nodes in the sequence share s edges. If s=1 these are the
		path components of the hypergraph. 

		Example		
		-------
			>>> S = {'A':{1,2,3},'B':{2,3,4},'C':{5,6},'D':{6}}
			>>> H = Hypergraph(S)

			>>> list(H.s_components(edges=True))
			[{'C', 'D'}, {'A', 'B'}]
			>>> list(H.s_components(edges=False))
			[{1, 2, 3, 4}, {5, 6}]

		"""

		if edges:
			A,coldict = self.edge_adjacency_matrix(s=s,index=True)
			G = nx.from_scipy_sparse_matrix(A)
			for c in nx.connected_components(G):
				yield {coldict[e] for e in c}
		else:
			A,rowdict = self.adjacency_matrix(s=s,index=True)
			G = nx.from_scipy_sparse_matrix(A)
			for c in nx.connected_components(G):
				yield {rowdict[n] for n in c}

	def s_component_subgraphs(self,s=1,edges=True):
		"""
		Returns a generator for the induced subgraphs of s_connected components.

		Parameters
		----------
		s : int, optional, default: 1

		edges: boolean, optional, edges=False
			Determines if edge or node components are desired. Returns 
			subgraphs equal to the hypergraph restricted to each set of nodes(edges) in the 
			s-connected components or s-edge-connected components

		Returns
		-------
		s_component_subgraphs : iterator
			Iterator returns subgraphs generated by the edges (or nodes) in the 
			s-edge(node) components of hypergraph. 

		"""
		for idx,c in enumerate(self.s_components(s=s,edges=edges)):
			if edges:
				yield self.restrict_to_edges(c,name=f'{self.name}:{idx}')
			else:
				yield self.restrict_to_nodes(c,name=f'{self.name}:{idx}')

	def s_components(self,s=1,edges=True):
		"""
		Same as s_connected_components
		"""
		return self.s_connected_components(s=s,edges=edges)

	def connected_components(self, edges=False):
		"""
		Same as :meth:`s_connected_components` with s=1.
		"""
		return self.s_connected_components(edges=edges)

	def connected_component_subgraphs(self, edges=False):
		"""
		Same as :meth:`s_component_subgraphs` with s=1
		"""
		return self.s_component_subgraphs(edges=edges)


	def components(self, edges=False):
		"""
		Same as :meth:`s_connected_components` with s=1
		"""
		return self.s_components(s=1,edges=edges)

	def component_subgraphs(self, edges=False):
		"""
		Same as :meth:`s_components_subgraphs` with s=1
		"""
		return self.s_component_subgraphs(edges=False)

	def node_diameters(self,s=1):
		"""
		Returns the node diameters of the connected components in hypergraph.

		Parameters
		----------


		Returns:
		an array of the diameters of the s-components and 
		an array of the s-component nodes.
		"""
		A,coldict = self.adjacency_matrix(s=s, index=True)
		G = nx.from_scipy_sparse_matrix(A) 
		diams = []
		comps = []
		for c in nx.connected_components(G):
		    diamc = nx.diameter(G.subgraph(c))
		    temp = set()
		    for e in c:
		    	temp.add(coldict[e])
		    comps.append(temp)
		    diams.append(diamc)
		loc = np.argmax(diams)
		return diams[loc], diams, comps

	def edge_diameters(self,s=1):
		"""
		Returns the edge diameters of the s_edge_connected component subgraphs 
		in hypergraph.
		
		Parameters
		----------
		s : int, optional, default: 1

		Returns
		-------
		maximum diameter : int

		list of diameters : list
			List of edge_diameters for s-edge component subgraphs in hypergraph

		list of component : list
			List of the edge uids in the s-edge component subgraphs.

		"""
		A,coldict = self.edge_adjacency_matrix(s=s, index=True)
		G = nx.from_scipy_sparse_matrix(A) 
		diams = []
		comps = []
		for c in nx.connected_components(G):
		    diamc = nx.diameter(G.subgraph(c))
		    temp = set()
		    for e in c:
		    	temp.add(coldict[e])
		    comps.append(temp)
		    diams.append(diamc)
		loc = np.argmax(diams)
		return diams[loc], diams, comps

	def diameter(self,s=1):
		"""
		Returns the length of the longest shortest s-walk between nodes in hypergraph

		Parameters
		----------
		s : int, optional, default: 1

		Returns
		-------
		diameter : int

		Raises
		------
		HyperNetXError
			If hypergraph is not s-edge-connected

		Notes
		-----
		Two nodes are s-adjacent if they share s edges.
		Two nodes v_start and v_end are s-walk connected if there is a sequence of
		nodes v_start, v_1, v_2, ... v_n-1, v_end such that consecutive nodes
		are s-adjacent. If the graph is not connected, an error will be raised.

		"""
		A = self.adjacency_matrix(s=s)
		G = nx.from_scipy_sparse_matrix(A)
		if nx.is_connected(G):
			return nx.diameter(G)
		else:
			raise HyperNetXError(f'Hypergraph is not s-connected. s={s}')

	def edge_diameter(self,s=1):
		"""
		Returns the length of the longest shortest s-walk between edges in hypergraph
		
		Parameters
		----------
		s : int, optional, default: 1

		Return
		------
		edge_diameter : int

		Raises
		------
		HyperNetXError
			If hypergraph is not s-edge-connected

		Notes
		-----
		Two edges are s-adjacent if they share s nodes.
		Two nodes e_start and e_end are s-walk connected if there is a sequence of
		edges e_start, e_1, e_2, ... e_n-1, e_end such that consecutive edges
		are s-adjacent. If the graph is not connected, an error will be raised.

		"""
		A = self.edge_adjacency_matrix(s=s)
		G = nx.from_scipy_sparse_matrix(A)
		if nx.is_connected(G):
			return nx.diameter(G)
		else:
			raise HyperNetXError(f'Hypergraph is not s-connected. s={s}')


	def distance(self,source,target,s=1):
		"""
		Returns the shortest s-walk distance between two nodes in the hypergraph.

		Parameters
		----------  
		source : node.uid or node
			a node in the hypergraph   

		target : node.uid or node
			a node in the hypergraph  

		s : positive integer 
			the number of edges

		Returns
		-------
		s-walk distance : int

		See Also
		--------
		edge_distance

		Notes
		-----
		The s-distance is the shortest s-walk length between the nodes.
		An s-walk between nodes is a sequence of nodes that pairwise share
		at least s edges. The length of the shortest s-walk is 1 less than
		the number of nodes in the path sequence.

		Uses the networkx shortest_path_length method on the graph
		generated by the s-adjacency matrix.

		"""
		if isinstance(source,Entity):
			source = source.uid
		if isinstance(target,Entity):
			target = target.uid
		A,rowdict = self.adjacency_matrix(s=s,index=True)
		g = nx.from_scipy_sparse_matrix(A)
		rkey = {v:k for k,v in rowdict.items()}
		try:
			path = nx.shortest_path_length(g,rkey[source],rkey[target])
			return path
		except:
			warnings.warn(f'No {s}-path between {source} and {target}')
			return np.inf

	def edge_distance(self,source,target,s=1):
		"""
		Returns the shortest s-walk distance between two edges in the hypergraph.

		Parameters
		----------  
		source : edge.uid or edge
			an edge in the hypergraph 

		target : edge.uid or edge
			an edge in the hypergraph  

		s : positive integer 
			the number of intersections between pairwise consecutive edges

		Returns
		-------
		s-walk distance : the shortest s-walk edge distance
			A shortest s-walk is computed as a sequence of edges,
			the s-walk distance is the number of edges in the sequence
			minus 1. If no such path exists returns np.inf.

		See Also
		--------
		distance

		Notes
		-----
			The s-distance is the shortest s-walk length between the edges.
			An s-walk between edges is a sequence of edges such that consecutive pairwise 
			edges intersect in at least s nodes. The length of the shortest s-walk is 1 less than
			the number of edges in the path sequence.

			Uses the networkx shortest_path_length method on the graph
			generated by the s-edge_adjacency matrix.

		"""
		if isinstance(source,Entity):
			source = source.uid
		if isinstance(target,Entity):
			target = target.uid
		A,coldict = self.edge_adjacency_matrix(s=s,index=True)
		g = nx.from_scipy_sparse_matrix(A)
		ckey = {v:k for k,v in coldict.items()}
		try:
			path =  nx.shortest_path_length(g,ckey[source],ckey[target])
			return path
		except:
			warnings.warn(f'No {s}-path between {source} and {target}')
			return np.inf

	@classmethod
	def from_bipartite(cls,B,set_names=[0,1],name=None):
		"""
		Static method creates a Hypergraph from a bipartite graph.

		Parameters
		----------

		B: nx.Graph()
			A networkx bipartite graph. Each node in the graph has a property
			'bipartite' taking one of two values

		set_names: iterable
			An ordered list :math:`[x_0, x_1]`, corresponding to the values 
			assigned to the bipartite property in B.

		name: hashable

		Returns
		-------
		new hypergraph : Hypergraph

		Notes
		-----
		A partition for the nodes in a bipartite graph generates a hypergraph as follows.
		For each node n in B with bipartite property equal to set_names[0] there is a 
		node n in the hypergraph.  For each node e in B with bipartite property
		equal to set_names[1], there is an edge in the hypergraph. 
		For each edge (n,e) in B add n to the edge e in the hypergraph.

		""" 

		if not bipartite.is_bipartite(B):
			raise HyperNetxError('Error: Method requires a bipartite graph.')
		entities = []
		for n,d in B.nodes(data=True):
		    if d['bipartite'] == set_names[1]:
		        elements = []
		        for nei in B.neighbors(n):
		            elements.append(Entity(nei,[],properties=B.nodes(data=True)[nei]))
		        if elements:
		            entities.append(Entity(n,elements,properties=d))
		name = name or '_'
		return Hypergraph(EntitySet(name,entities),name=name)


	@classmethod
	def from_numpy_array(cls,M,node_names=None, edge_names=None, name=None, key=None):
		"""
		Create a hypergraph from a real valued matrix represented as a numpy array with dimensions 2x2 
		The matrix is converted to a matrix of 0's and 1's so that any truthy cells are converted to 1's and 
		all others to 0's. 

		Parameters
		----------
		M : real valued array-like object, dimensions=2x2
		    representing a real valued matrix with rows corresponding to nodes and columns to edges

		node_names : object, array-like, default=None
		    List of node names must be the same length as M.shape[0]. 
		    If None then the node names correspond to row indices with 'v' prepended.

		edge_names : object, array-like, default=None
		    List of edge names must have the same length as M.shape[1]. 
		    If None then the edge names correspond to column indices with 'e' prepended.

		name : hashable

		key : (optional) function
			boolean function to be evaluated on each cell of the array
		    
		Returns
		-------
		 : Hypergraph

		Note
		----
		The constructor does not generate empty edges. 
		All zero columns in M are removed and the names corresponding to these
		edges are discarded.


		"""  	    
		## Create names for nodes and edges
		## Validate the size of the node and edge arrays

		M = np.array(M)
		if len(M.shape) != (2):
			raise HyperNetXError('Input requires a 2 dimensional numpy array')

		if node_names is not None:
		    nodenames = np.array(node_names)
		    if len(nodenames) != M.shape[0]:
		        raise HyperNetXError('Number of node names does not match number of rows.')
		else:
		    nodenames = np.array([f'v{idx}' for idx in range(M.shape[0])])

		if edge_names is not None:
		    edgenames = np.array(edge_names)
		    if len(edgenames) != M.shape[1]:
		    	raise HyperNetXError('Number of edge_names does not match number of columns.')
		else:
		    edgenames = np.array([f'e{jdx}' for jdx in range(M.shape[1])])

		## apply boolean key if available
		if key:
			M = key(M)

		## Remove empty column indices from M columns and edgenames
		colidx = np.array([jdx for jdx in range(M.shape[1]) if any(M[:,jdx])])
		colidxsum = np.sum(colidx)
		if not colidxsum:
			return Hypergraph()
		else:
			M = M[:,colidx]
			edgenames = edgenames[colidx]
			edict = dict()
			## Create an EntitySet of edges from M         
			for jdx,e in enumerate(edgenames):
			    edict[e] = nodenames[[idx for idx in range(M.shape[0]) if M[idx,jdx]]]		            
			return Hypergraph(edict,name=name)


	@classmethod
	def from_dataframe(cls, df, fillna=0, transpose=False, name=None, key=None):
	    '''
	    Create a hypergraph from a Pandas Dataframe object using index to label vertices
	    and Columns to label edges. 

	    Parameters
	    ----------
	    df : Pandas.Dataframe
	        a real valued dataframe with a single index

	    fillna : float, default = 0
	        a real value to place in empty cell, all-zero columns will not generate
	        an edge

	    transpose : bool, default = False
	        option to transpose the dataframe, in this case df.Index will label the edges
	        and df.columns will label the nodes

	    key : (optional) function
			boolean function to be evaluated on each cell of the array
	        
	    Returns
	    -------
	    : Hypergraph

	    Note
	    ----
	    The constructor does not generate empty edges. 
	    All-zero columns in df are removed and the names corresponding to these
	    edges are discarded.    
	    '''
	    import pandas as pd

	    if type(df) != pd.core.frame.DataFrame:
	        raise HyperNetXError('Error: Input object must be a pandas dataframe.') 
	    if transpose:
	        df = df.transpose()   
	    node_names = np.array(df.index)
	    edge_names = np.array(df.columns)
	    df = df.fillna(fillna)
	    if key:
	    	df = df.apply(key)
	    return cls.from_numpy_array(df.values,node_names=node_names,edge_names=edge_names,name=name)





