# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.
import warnings
import pickle
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.assortativity import neighbor_degree
import numpy as np
import pandas as pd
from scipy.sparse import issparse, coo_matrix, dok_matrix, csr_matrix
from collections import OrderedDict, defaultdict
from hypernetx.classes.entity import Entity, EntitySet
from hypernetx.classes.staticentity import StaticEntity, StaticEntitySet
from hypernetx.exception import HyperNetXError
from hypernetx.utils.decorators import not_implemented_for
from hypernetx.utils.extras import HNXCount, DefaultOrderedDict
from collections import OrderedDict

__all__ = ["Hypergraph"]


class Hypergraph:
    """
    Hypergraph H = (V,E) references a pair of disjoint sets:
    V = nodes (vertices) and E = (hyper)edges.

    An HNX Hypergraph is either dynamic or static.
    Dynamic hypergraphs can change by adding or subtracting objects
    from them. Static hypergraphs require that all of the nodes and edges
    be known at creation. A hypergraph is dynamic by default.

    *Dynamic hypergraphs* require the user to keep track of its objects,
    by using a unique name for each node in the list of nodes and each edge in the list of edges, though node names may overlap with edge names.
    This allows for multi-edge graphs and inseparable nodes.
    If the user does not specify the uids of each object, they will be created automatically.

    For example: Let V = {1,2,3} and E = {e1,e2,e3},
    where e1 = {1,2}, e2 = {1,2}, and e3 = {1,2,3}.
    The edges e1 and e2 contain the same set of nodes and yet
    are distinct and must be distinguishable within H.

    In a dynamic hypergraph each node and edge is
    instantiated as an Entity and given an identifier or uid. Entities
    keep track of connections with other entities with their "elements" property. Since
    hypergraphs can be quite large, only the entity identifiers will be used
    for computation intensive methods, this means the user must take care
    to keep a one to one correspondence between their set of uids and
    the objects in their hypergraph. See `Honor System`_

    *Static hypergraphs* store node and edge information in numpy arrays and
    are immutable. Each node and edge receives a class generated internal
    identifier used for computations so do not require the user to create
    different ids for nodes and edges. To create a static hypergraph set
    `static = True` in the signature.

    We will create hypergraphs in multiple ways:

    1. As an empty instance: ::

        >>> H = hnx.Hypergraph()
        >>> H.nodes, H.edges
        ({}, {})

    2. From a dictionary of iterables (elements of iterables must be of type hypernetx.Entity or hashable): ::

        >>> H = Hypergraph({'a':[1,2,3],'b':[4,5,6]})
        >>> H.nodes, H.edges
        # output: (EntitySet(_:Nodes,[1, 2, 3, 4, 5, 6],{}), EntitySet(_:Edges,['b', 'a'],{}))

    3. From an iterable of iterables: (elements of iterables must be of type hypernetx.Entity or hashable): ::

        >>> H = Hypergraph([{'a','b'},{'b','c'},{'a','c','d'}])
        >>> H.nodes, H.edges
        # output: (EntitySet(_:Nodes,['d', 'b', 'c', 'a'],{}), EntitySet(_:Edges,['_1', '_2', '_0'],{}))

    4. From a hypernetx.EntitySet or StaticEntitySet: ::

        >>> a = Entity('a',{1,2}); b = Entity('b',{2,3})
        >>> E = EntitySet('sample',elements=[a,b])
        >>> H = Hypergraph(E)
        >>> H.nodes, H.edges.
        # output: (EntitySet(_:Nodes,[1, 2, 3],{}), EntitySet(_:Edges,['b', 'a'],{}))

    All of these constructions apply for both dynamic and static hypergraphs. To
    create a static hypergraph set the parameter `static=True`. In addition a static
    hypergraph is automatically created if a StaticEntity, StaticEntitySet, or pandas.DataFrame object
    is passed to the Hypergraph constructor.

    5. | From a pandas.DataFrame. The dataframe must have at least two columns with headers and there can be no nans. 
       | By default the first column corresponds to the edge names and the second column to the node names.
       | You can specify the columns by restricting the dataframe to the columns of interest in the order:
       | :code:`hnx.Hypergraph(df[[edge_column_name,node_column_name]])`
       | See :ref:`Colab Tutorials <colab>`  Tutorial 6 - Static Hypergraphs and Entities for additional information.


    Parameters
    ----------
    setsystem : (optional) EntitySet, StaticEntitySet, dict, iterable, pandas.dataframe, default: None
        See notes above for setsystem requirements.

    name : hashable, optional, default: None
        If None then a placeholder '_'  will be inserted as name

    static : boolean, optional, default: False
        If True the hypergraph will be immutable, edges and nodes may not be changed.

    use_nwhy : boolean, optional, default = False
        If True hypergraph will be static and computations will be done using
        C++ backend offered by NWHypergraph. This requires installation of the
        NWHypergraph C++ library. Please see the :ref:`NWHy documentation <nwhy>` for more information.


    """

    # TODO: remove lambda functions from constructor in H and E.

    def __init__(
        self, setsystem=None, name=None, static=False, use_nwhy=False, filepath=None
    ):        
        self.filepath = filepath
        if use_nwhy:
            static = True
            try:
                import nwhy

                self.nwhy = True

            except:
                self.nwhy = False
                print("NWHypergraph is not available. Will continue with static=True.")
                use_nwhy = False
        else:
            self.nwhy = False
        if not name:
            self.name = ""
        else:
            self.name = name

        if static == True or (
            isinstance(setsystem, StaticEntitySet) or
            isinstance(setsystem, StaticEntity) or
            isinstance(setsystem, pd.DataFrame) ) :
            self._static=True
            if setsystem is None:
                self._edges=StaticEntitySet()
                self._nodes=StaticEntitySet()
            else:
                E=StaticEntitySet(entity = setsystem)
                self._edges=E
                self._nodes=E.restrict_to_levels([1])
        else:
            self._static=False
            if setsystem is None:
                setsystem=[]
            # the constructor for EntitySet can accept both a list of iterables and a dictionary. We skip constructing an entityset if there already is one.
            if not isinstance(setsystem, EntitySet):
                # initialize an entityset object
                setsystem = EntitySet(f"{self.name}:Edges", elements=setsystem)

            self._edges = setsystem
            self._nodes = EntitySet(f"{self.name}:Nodes", elements=self._edges.get_dual())

        if self._static:
            temprows, tempcols = self.edges.data.T
            tempdata = np.ones(len(temprows), dtype=int)
            self.state_dict = {
                "data": (temprows, tempcols, tempdata)
            }  # how can we incorporate the counts into the nwhy hypergraph?
            if self.nwhy:
                self.g = nwhy.NWHypergraph(*self.state_dict["data"])
                self.nwhy_dict = {"snodelg": dict(), "sedgelg": dict()}
            self.state_dict["snodelg"] = dict()
            self.state_dict["sedgelg"] = dict()
            if self.filepath is not None:
                self.save_state(fpath=self.filepath)
    @property
    def edges(self):
        """
        Object associated with self._edges.
        
        Returns
        -------
        StaticEntitySet or EntitySet
            If self.isstatic the StaticEntitySet, otherwise EntitySet.
        """
        return self._edges

    @property
    def nodes(self):
        """
        Object associated with self._nodes.
        
        Returns
        -------
        StaticEntitySet or EntitySet
            If self.isstatic the StaticEntitySet, otherwise EntitySet.

        """
        return self._nodes

    @property
    def isstatic(self):
        """
        Checks whether nodes and edges are immutable
        
        Returns
        -------
        Boolean
    
        """
        return self._static

    @property
    def incidence_dict(self):
        """
        Dictionary keyed by edge uids with values the uids of nodes in each edge
        
        Returns
        -------
        dict
        
        """
        return self._edges.incidence_dict

    @property
    def shape(self):
        """
        (number of nodes, number of edges)
        
        Returns
        -------
        tuple
            
        """
        if self.nwhy:
            return (self.g.number_of_nodes(), self.g.number_of_edges())
        else:
            return (len(self._nodes.elements), len(self._edges.elements))

    def __str__(self):
        """
        String representation of hypergraph
        
        Returns
        -------
        str
            
        """
        return f"Hypergraph({self.edges.elements},name={self.name})"

    def __repr__(self):
        """
        String representation of hypergraph
        
        Returns
        -------
        str
            
        """
        return f"Hypergraph({self.edges.elements},name={self.name})"

    def __len__(self):
        """
        Number of nodes
        
        Returns
        -------
        int
            
        """
        if self.nwhy:
            return self.g.number_of_nodes()
        else:
            return len(self._nodes)

    def __iter__(self):
        """
        Iterate over the nodes of the hypergraph
        
        Returns
        -------
        dict_keyiterator
            
        """
        return iter(self.nodes)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.nodes

        Parameters
        ----------
        item : hashable or Entity

        """
        if isinstance(item, Entity):
            return item.uid in self.nodes
        else:
            return item in self.nodes

    def __getitem__(self, node):
        """
        Returns the neighbors of node

        Parameters
        ----------
        node : Entity or hashable
            If hashable, then must be uid of node in hypergraph

        Returns
        -------
        neighbors(node) : iterator

        """
        return self.neighbors(node)

    @not_implemented_for("dynamic")
    def get_id(self, uid, edges=False):
        """
        Return the internally assigned id associated with a label.

        Parameters
        ----------
        uid : string
            User provided name/id/label for hypergraph object
        edges : bool, optional
            Determines if uid is an edge or node name
        
        Returns
        -------
        : int
            internal id assigned at construction
        """
        kdx = (edges + 1) % 2
        return int(np.argwhere(self.edges.labs(kdx) == uid)[0])

    @not_implemented_for("dynamic")
    def get_name(self, id, edges=False):
        """
        Return the user defined name/id/label associated to an
        internally assigned id.

        Parameters
        ----------
        id : int
            Internally assigned id
        edges : bool, optional
            Determines if id references an edge or node
        
        Returns
        -------
        str
            User provided name/id/label for hypergraph object
        """
        kdx = (edges + 1) % 2
        return self.edges.labs(kdx)[id]

    @not_implemented_for("dynamic")
    def get_linegraph(self, s, edges=True, use_nwhy=True):
        """
        Creates an ::term::s-linegraph for the Hypergraph.
        If edges=True (default)then the edges will be the vertices of the line graph.
        Two vertices are connected by an s-line-graph edge if the corresponding
        hypergraphedges intersect in at least s hypergraph nodes.
        If edges=False, the hypergraph nodes will be the vertices of the line graph.
        Two vertices are connected if the nodes they correspond to share at least s
        incident hyper edges.

        Parameters
        ----------
        s : int
            The width of the connections.
        edges : bool, optional
            Determine if edges or nodes will be the vertices in the linegraph.
        use_nwhy : bool, optional
            Requests that nwhy be used to construct the linegraph. If NWHy is not available this is ignored.

        Returns
        -------
        nx.Graph
            A NetworkX graph.
        """
        if use_nwhy and self.nwhy:
            d = self.nwhy_dict
        else:
            d = self.state_dict
        key = "sedgelg" if edges else "snodelg"
        if s in d[key]:
            return d[key][s]
        else:
            if use_nwhy and self.nwhy:
                d[key][s] = self.g.s_linegraph(s=s, edges=edges)
            else:
                if edges:
                    A = self.edge_adjacency_matrix(s=s)
                else:
                    A = self.adjacency_matrix(s=s)
                d[key][s] = nx.from_scipy_sparse_matrix(A)
                if self.filepath is not None:
                    self.save_state(fpath=self.filepath)
            return d[key][s]

    @not_implemented_for("dynamic")
    def set_state(self, **kwargs):
        """
        Allow state_dict updates from outside of class. Use with caution.

        Parameters
        ----------
        **kwargs
            key=value pairs to save in state dictionary
        """
        self.state_dict.update(kwargs)
        if self.filepath is not None:
            self.save_state(fpath=self.filepath)

    @not_implemented_for("dynamic")
    def save_state(self, fpath=None):
        """
        Save the hypergraph as an ordered pair: [state_dict,labels]
        The hypergraph can be recovered using the command:

            >>> H = hnx.Hypergraph.recover_from_state(fpath)

        Parameters
        ----------
        fpath : str, optional
        """
        if fpath is None:
            fpath = self.filepath or "current_state.p"
        pickle.dump([self.state_dict, self.labels], open(fpath, "wb"))

    @classmethod
    def recover_from_state(cls, fpath="current_state.p", newfpath=None, use_nwhy=True):
        """
        Recover a static hypergraph pickled using save_state.

        Parameters
        ----------
        fpath : str
            Full path to pickle file containing state_dict and labels
            of hypergraph

        Returns
        -------
        H : Hypergraph
            static hypergraph with state dictionary prefilled
        """
        temp, labels = pickle.load(open(fpath, "rb"))
        recovered_data = np.array(temp["data"])[[0, 1]].T  # need to save counts as well
        recovered_counts = np.array(temp["data"])[
            [2]
        ]  # ammend this to store cell weights
        E = StaticEntitySet(data=recovered_data, labels=labels)
        E.properties["counts"] = recovered_counts
        H = Hypergraph(E, use_nwhy=use_nwhy)
        H.state_dict.update(temp)
        if newfpath == "same":
            newfpath = fpath
        if newfpath is not None:
            H.filepath = newfpath
            H.save_state()
        return H

    @classmethod
    def add_nwhy(cls, h, fpath=None):
        """
        Add nwhy functionality to a hypergraph.

        Parameters
        ----------
        h : hnx.Hypergraph
        fpath : file path for storage of hypergraph state dictionary

        Returns
        -------
        hnx.Hypergraph
            Returns a copy of h with static set to true and nwhy set to True
            if it is available.

        """

        if h.isstatic:
            sd = h.state_dict
            H = Hypergraph(h.edges, use_nwhy=True, filepath=fpath)
            H.state_dict.update(sd)
            return H
        else:
            return Hypergraph(StaticEntitySet(h.edges), use_nwhy=True, filepath=fpath)

    def edge_size_dist(self):
        """
        Returns the size for each edge
        
        Returns
        -------
        np.array
            
        """
        if self.isstatic:
            dist = self.state_dict.get("edge_size_dist", None)
            if dist:
                return dist
            else:
                if self.nwhy:
                    dist = self.g.edge_size_dist()
                else:
                    dist = list(np.array(np.sum(self.incidence_matrix(), axis=0))[0])

                self.set_state(edge_size_dist=dist)
                return dist
        else:
            # return list(np.array(np.sum(self.incidence_matrix(), axis=0))[0])
            return [len(self.edges[id]._elements) for id in self.edges]#np.array(np.sum(self.incidence_matrix(), axis=0))[0])

    def convert_to_static(
        self,
        name=None,
        nodes_name="nodes",
        edges_name="edges",
        use_nwhy=False,
        filepath=None,
    ):
        """
        Returns new static hypergraph with the same dictionary as original hypergraph

        Parameters
        ----------
        name : None, optional
            Name
        nodes_name : str, optional
            name for list of node labels
        edges_name : str, optional
            name for list of edge labels

        Returns
        -------
        hnx.Hypergraph
            Will have attribute static = True

        Note
        ----
        Static hypergraphs store the user defined node and edge names in
        a dictionary of labeled lists. The order of the lists provides an
        index, which the hypergraph uses in place of the node and edge names
        for fast processing.
        """
        arr, cdict, rdict = self.edges.incidence_matrix(index=True)
        labels = OrderedDict(
            [
                (edges_name, [cdict[k] for k in range(len(cdict))]),
                (nodes_name, [rdict[k] for k in range(len(rdict))]),
            ]
        )
        E = StaticEntity(arr=arr.T, labels=labels)
        return Hypergraph(setsystem=E, name=name)

    def remove_static(self, name=None):
        """
        Returns dynamic hypergraph

        Parameters
        ----------
        name : None, optional
            User defined namae of hypergraph

        Returns
        -------
        hnx.Hypergraph
            A new hypergraph with the same dictionary as self but allowing dynamic
            changes to nodes and edges.
            If hypergraph is not static, returns self.
        """
        if not self.isstatic:
            return self
        else:
            return Hypergraph(self.edges.incidence_dict, name=name)

    def translate(self, idx, edges=False):
        """
        Returns the translation of numeric values associated with hypergraph.
        Only needed if exposing the static identifiers assigned by the class.
        If not static then the idx is returned.

        Parameters
        ----------
        idx : int
            class assigned integer for internal manipulation of Hypergraph data
        edges : bool, optional, default: True
            If True then translates from edge index. Otherwise will translate from
            node index, default=False

        Returns
        -------
         : int or string
            User assigned identifier corresponding to idx
        """
        if self.isstatic:
            return self.get_name(idx, edges=edges)
        else:
            return idx

    def degree(self, node, s=1, max_size=None):
        """
        The number of edges of size s that contain node.

        Parameters
        ----------
        node : hashable
            identifier for the node.
        s : positive integer, optional, default: 1
            smallest size of edge to consider in degree
        max_size : positive integer or None, optional, default: None
            largest size of edge to consider in degree

        Returns
        -------
         : int

        """
        if self.isstatic:
            ndx = self.get_id(node)
            #             if s == 1:
            #                 return np.sum(self.edges.data.T[1] == ndx)
            if self.nwhy:
                return self.g.degree(ndx, min_size=s, max_size=None)

            else:
                if max_size is not None:
                    ids = np.where(
                        np.array(self.edge_size_dist()) in range(s, max_size + 1)
                    )[0]
                else:
                    ids = np.where(np.array(self.edge_size_dist()) >= s)[0]
                imat = self.incidence_matrix()
                return np.sum(imat[ndx, ids])
        else:
            if max_size is not None:
                memberships = set(self._nodes[node]._elements)
                return len(
                    set(e for e in memberships if len(self.edges[e]) in range(s, max_size + 1))
                )
            elif s > 1:
                memberships = set(self._nodes[node]._elements)
                return len(set(e for e in memberships if len(self._edges[e]) >= s))
            else:
                return self._nodes[node].size

    def size(self, edge):
        """
        The number of nodes that belong to edge.

        Parameters
        ----------
        edge : hashable
            The uid of an edge in the hypergraph

        Returns
        -------
        size : int

        """
        if self.isstatic:
            edx = self.get_id(edge, edges=True)
            esd = self.state_dict.get("edge_size_dist", None)
            if esd is not None:
                return esd[edx]
            else:
                if self.nwhy:
                    return self.g.size(edx)
                else:
                    return np.sum(self.edges.data.T[0] == edx)
        else:
            return self.edges[edge].size

    def number_of_nodes(self, nodeset=None):
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
            if self.nwhy == True:
                return self.g.number_of_nodes()
            else:
                return len(self.nodes)

    def number_of_edges(self, edgeset=None):
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
            if self.nwhy == True:
                return self.g.number_of_edges()
            else:
                return len(self.edges)

    def order(self):
        """
        The number of nodes in hypergraph.

        Returns
        -------
        order : int
        """
        if self.nwhy:
            return self.g.number_of_nodes()
        else:
            return len(self.nodes)

    def dim(self, edge):
        """
        Same as size(edge)-1.
        """
        return self.size(edge) - 1

    def neighbors(self, node, s=1):
        """
        The nodes in hypergraph which share s edge(s) with node.

        Parameters
        ----------
        node : hashable or Entity
            uid for a node in hypergraph or the node Entity

        s : int, list, optional, default : 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
         : list
            List of neighbors

        """
        if not node in self.nodes:
            print(f"Node is not in hypergraph {self.name}.")
            return

        if self.isstatic:
            g = self.get_linegraph(s=s, edges=False)
            ndx = self.get_id(node)
            if self.nwhy == True:
                nbrs = g.s_neighbors(ndx)
            else:
                nbrs = list(g.neighbors(ndx))
            return [self.translate(nb, edges=False) for nb in nbrs]

        else:
            neighbors = defaultdict(lambda : 0)
            for e in self.nodes[node]._elements:
                for nbr in self.edges[e]._elements:
                    if nbr != node:
                        neighbors[nbr] += 1
            if s == 1:
                return list(neighbors.keys())
            else:
                return [key for key, val in neighbors.items() if val >= s]

    def edge_neighbors(self, edge, s=1):
        """
        The edges in hypergraph which share s nodes(s) with edge.

        Parameters
        ----------
        edge : hashable or Entity
            uid for a edge in hypergraph or the edge Entity

        s : int, list, optional, default : 1
            Minimum number of nodes shared by neighbors edge node.

        Returns
        -------
         : list
            List of edge neighbors

        """
        if not edge in self.edges:
            print(f"Edge is not in hypergraph {self.name}.")
            return

        if self.isstatic:
            g = self.get_linegraph(s=s, edges=True)
            edx = self.get_id(edge, edges=True)
            if self.nwhy == True:
                nbrs = g.s_neighbors(edx)
            else:
                nbrs = list(g.neighbors(edx))
            return [self.translate(nb, edges=True) for nb in nbrs]

        else:
            node = self.edges[edge].uid
            return self.dual().neighbors(node, s=s)

    @not_implemented_for("static")
    def remove_node(self, node):
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
            if not isinstance(node, Entity):
                node = self._nodes[node]
            for edge in node._elements:
                self._edges[edge]._remove(node)
            self._nodes._remove(node)
        return self

    @not_implemented_for("static")
    def remove_nodes(self, node_set):
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

    # @not_implemented_for("static")
    # def add_node(self, node):
    #     """

    #     Adds a single node to hypergraph.

    #     Parameters
    #     ----------
    #     node : hashable or Entity
    #         If hashable the node's elements will be an empty list.

    #     Returns
    #     -------
    #     hypergraph : Hypergraph

    #     Notes
    #     -----

    #     """
    #     if node in self._nodes:
    #         warnings.warn("Cannot add edge. Edge already in hypergraph")
    #     elif isinstance(node, Entity):
    #         if len(node) > 0:
    #             self._nodes.add_element(Entity(node.uid, elements=node.elements, **node.properties))
    #             for e in node.elements:
    #                 if e not in self._edges:
    #                     raise HyperNetXError("Node references a non-existent edge.")
    #                 self._edges[e].add(node.uid)
    #         else:
    #             self._nodes.add_element(Entity(node.uid, **node.properties))
    #     else:
    #         self._nodes.add_element(Entity(node))  # this generates an empty node with a user-defined uid.
    #     return self

    @not_implemented_for("static")
    def _add_nodes_from(self, nodes):
        """
        Instantiates new nodes.

        Parameters
        ----------
        nodes : iterable of hashables or Entities

        """
        for node in nodes:
            if node in self._nodes and isinstance(node, Entity):
                self._nodes[node].__dict__.update(node.properties)
            elif node not in self._nodes:
                if isinstance(node, Entity):
                    self._nodes._add_element(Entity(node.uid, node._elements, **node.properties))
                else:
                    self._nodes._add_element(Entity(node))

    @not_implemented_for("static")
    def add_edge(self, edge):
        """

        Adds a single edge to hypergraph.

        Parameters
        ----------
        edge : hashable, iterable, or Entity
            If hashable, we assume that it's the uid and the edge created will be empty.
            If iterable, the uid is automatically assigned and the iterable is cast to a list of the membership

        Returns
        -------
        hypergraph : Hypergraph

        Notes
        -----
        If a node contained in an edge is not present in the hypergraph, it is added and the hyperedge is added to the node's membership.
        Each node (element of edge) must be instantiated as a node,
        making sure its uid isn't already present in the self.

        """
        if edge in self._edges:
            warnings.warn("Cannot add edge. Edge already in hypergraph")
        elif isinstance(edge, Entity):
            if len(edge) > 0:
                self._add_nodes_from(edge._elements)
                self._edges._add_element(Entity(edge.uid, elements=edge._elements, **edge.properties))
                for n in edge._elements:
                    self._nodes[n]._add(edge.uid)
            else:
                self._edges._add_element(Entity(edge.uid, **edge.properties))# this generates an edge without elements
        else:
            try:
                # try to cast to a list
                edge = list(edge)
                uid = self._edges._add_element(edge) # if edge is an iterable
                self._add_nodes_from(edge)
                for n in edge:
                    self._nodes[n]._add(uid)
            except:
                # if this fails, we assume that it's a uid
                self._edges._add_element(Entity(edge)) # add an empty edge
        return self

    @not_implemented_for("static")
    def add_edges_from(self, edge_set):
        """
        Add edges to hypergraph.

        Parameters
        ----------
        edge_set : iterable of hashables, iterables, or Entities
            For hashables the edges returned will be empty.

        Returns
        -------
        hypergraph : Hypergraph

        """
        for edge in edge_set:
            self.add_edge(edge)
        return self

    # @not_implemented_for("static")
    def add_node_to_edge(self, node, edge):
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
            if node in self._nodes:
                self._edges[edge]._add(node)
                self._nodes[node]._add(edge)
            else:
                if not isinstance(node, Entity):
                    node = Entity(node)
                else:
                    node = Entity(node.uid, node._elements, **node.properties)
                self._edges[edge]._add(node)
                self._nodes._add_element(node)
                self._nodes[node]._add(edge)
        return self

    @not_implemented_for("static")
    def remove_edge(self, edge):
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
            if not isinstance(edge, Entity):
                edge = self._edges[edge]
            for node in edge._elements:
                if len(self._nodes[node]) == 1:
                    self._nodes._remove(node)
                else:
                    self._nodes[node]._remove(edge.uid)
            self._edges._remove(edge)
        return self

    @not_implemented_for("static")
    def remove_edges(self, edge_set):
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

    def incidence_matrix(self, index=False, weight = lambda self, node, edge : 1):
        """
        An incidence matrix for the hypergraph indexed by nodes x edges.

        Parameters
        ----------
        index : boolean, optional, default False
            If True return will include a dictionary of node uid : row number
            and edge uid : column number

        weight : a lambda function with inputs self, node, and edge returns a weight in the incidence matrix given the uid of the node and edge.
        The default is to return 1 when a node/edge pair exist.

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row dictionary : dict
            Dictionary identifying rows with nodes

        column dictionary : dict
            Dictionary identifying columns with edges

        """
        if self.isstatic:
            mat = self.state_dict.get("incidence_matrix", None)
            if mat is None:
                mat = self.edges.incidence_matrix()
                self.state_dict["incidence_matrix"] = mat
            if index:
                rdict = dict(enumerate(self.edges.labs(1)))
                cdict = dict(enumerate(self.edges.labs(0)))
                return mat, rdict, cdict
            else:
                return mat

        else:
            return self.edges.incidence_matrix(index=index, weight=weight)

    @staticmethod
    def incidence_to_adjacency(M, s=1, weighted=True):
        """
        Helper method to obtain adjacency matrix from incidence matrix.

        Parameters
        ----------
        M : scipy.sparse.csr.csr_matrix

        s : int, optional, default: 1

        weighted : boolean, optional, default: True

        Returns
        -------
        a matrix : scipy.sparse.csr.csr_matrix

        """
        A = M.dot(M.transpose())
        if issparse(A):
            A.setdiag(0)
            B = (A >= s) * 1
            A = A.multiply(B)
        else:
            np.fill_diagonal(A, 0)
            B = (A >= s) * 1
            A = np.multiply(A, B)

        if not weighted:
            A = (A > 0) * 1
        return csr_matrix(A)

    def adjacency_matrix(self, index=False, s=1, weighted=True):
        """
        The sparse weighted :term:`s-adjacency matrix`

        Parameters
        ----------
        s : int, optional, default: 1

        index: boolean, optional, default: False
            if True, will return a rowdict of row to node uid

        weighted: boolean, optional, default: True


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
        M = self.incidence_matrix(index=index)
        if index:
            return Hypergraph.incidence_to_adjacency(M[0], s=s, weighted=weighted), M[1]
        else:
            return Hypergraph.incidence_to_adjacency(M, s=s, weighted=weighted)

    def edge_adjacency_matrix(self, index=False, s=1, weighted=True):
        """
        The weighted :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default: 1

        index: boolean, optional, default: False
            if True, will return a coldict of column to edge uid

        sparse: boolean, optional, default: True

        weighted: boolean, optional, default: True

        Returns
        -------
        edge_adjacency_matrix : scipy.sparse.csr.csr_matrix or numpy.ndarray

        column dictionary : dict

        Notes
        -----
        This is also the adjacency matrix for the line graph.
        Two edges are s-adjacent if they share at least s nodes.
        If index=True, returns a dictionary column_index:edge_uid

        """
        M = self.incidence_matrix(index=index)
        if index:
            return (
                Hypergraph.incidence_to_adjacency(
                    M[0].transpose(), s=s, weighted=weighted
                ),
                M[2],
            )
        else:
            return Hypergraph.incidence_to_adjacency(
                M.transpose(), s=s, weighted=weighted
            )

    def auxiliary_matrix(self, s=1, index=False):
        """
        The unweighted :term:`s-auxiliary matrix` for hypergraph

        Parameters
        ----------
        s : int
        index : bool, optional, default: False
            return a dictionary of labels for the rows of the matrix


        Returns
        -------
        auxiliary_matrix : scipy.sparse.csr.csr_matrix or numpy.ndarray
            Will return the same type of matrix as self.arr

        Notes
        -----
        Creates subgraph by restricting to edges of cardinality at least s.
        Returns the unweighted s-edge adjacency matrix for the subgraph.

        """

        H = Hypergraph([e for e in self.edges.__call__() if e.size >= s])
        return H.edge_adjacency_matrix(s=s, index=index, weighted=False)

    def bipartite(self):
        """
        Constructs the networkX bipartite graph associated to hypergraph.

        Returns
        -------
        bipartite : nx.Graph()

        Notes
        -----
        Creates a bipartite networkx graph from hypergraph.
        The nodes and (hyper)edges of hypergraph become the nodes of bipartite graph.
        For every (hyper)edge e in the hypergraph and node n in e there is an edge (n,e)
        in the graph.

        """
        B = nx.Graph()
        E = self.edges
        V = self.nodes
        B.add_nodes_from(E, bipartite=1)
        B.add_nodes_from(V, bipartite=0)
        B.add_edges_from([(v, e) for e in E for v in self.edges[e]._elements])
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
        if self.isstatic:
            E = self.edges.restrict_to_levels((1, 0))
            return Hypergraph(E, name=name, use_nwhy=self.nwhy)
        else:
            return Hypergraph(self._nodes, name=name)

    def _collapse_nwhy(self, edges, rec):
        """
        Helper method for collapsing nodes and edges when hypergraph
        is static and using nwhy

        Parameters
        ----------
        edges : bool
            Collapse the edges if True, otherwise the nodes
        rec : bool
            return the equivalence classes
        """

        if edges:
            d = self.g.collapse_edges(return_equivalence_class=rec)
        else:
            d = self.g.collapse_nodes(return_equivalence_class=rec)

        if rec:
            en = {
                self.get_name(
                    k, edges=edges
                ): f"{self.get_name(k,edges=edges)}:{len(v)}"
                for k, v in d.items()
            }
            ec = {
                f"{self.get_name(k,edges=edges)}:{len(v)}": {
                    self.get_name(vd, edges=edges) for vd in v
                }
                for k, v in d.items()
            }
        else:
            en = {
                self.get_name(
                    k, edges=edges
                ): f"{self.get_name(k,edges=edges)}:{v.pop()}"
                for k, v in d.items()
            }
            ec = {}
        lev = self.edges.keys[1-1*edges]
        E = self.edges.restrict_to_indices(sorted(d.keys()), level=1-1*edges)
        E.labels[str(lev)] = np.array([en[k] for k in E.labels[lev]])
        if rec:
            return E, ec
        else:
            return E

    def collapse_edges(
        self,
        name=None,
        use_reps=None,
        return_counts=None,
        return_equivalence_classes=False,
    ):
        """
        Constructs a new hypergraph gotten by identifying edges containing the same nodes

        Parameters
        ----------
        name : hashable, optional, default: None

        return_equivalence_classes: boolean, optional, default: False
            Returns a dictionary of edge equivalence classes keyed by frozen sets of nodes

        Returns
        -------
        new hypergraph : Hypergraph
            Equivalent edges are collapsed to a single edge named by a representative of the equivalent
            edges followed by a colon and the number of edges it represents.

        equivalence_classes : dict
            A dictionary keyed by representative edge names with values equal to the edges in
            its equivalence class

        Notes
        -----
        Two edges are identified if their respective elements are the same.
        Using this as an equivalence relation, the uids of the edges are partitioned into
        equivalence classes.

        A single edge from the collapsed edges followed by a colon and the number of elements
        in its equivalence class as uid for the new edge


        """
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps ane return_counts are no longer supported keyword arguments and will throw
            an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        if self.nwhy:
            temp = self._collapse_nwhy(True, return_equivalence_classes)
        else:
            temp = self.edges.collapse_identical_elements(
                "_", return_equivalence_classes=return_equivalence_classes
            )
        if return_equivalence_classes:
            return Hypergraph(temp[0], name, use_nwhy=self.nwhy), temp[1]
        else:
            return Hypergraph(temp, name, use_nwhy=self.nwhy)

    def collapse_nodes(
        self,
        name=None,
        use_reps=True,
        return_counts=True,
        return_equivalence_classes=False,
    ):
        """
        Constructs a new hypergraph gotten by identifying nodes contained by the same edges

        Parameters
        ----------
        name: str, optional, default: None

        return_equivalence_classes: boolean, optional, default: False
            Returns a dictionary of node equivalence classes keyed by frozen sets of edges

        use_reps : boolean, optional, default: False - Deprecated, this no longer works and will be removed
            Choose a single element from the collapsed nodes as uid for the new node, otherwise uses
            a frozen set of the uids of nodes in the equivalence class

        return_counts: boolean, - Deprecated, this no longer works and will be removed
            if use_reps is True the new nodes have uids given by a tuple of the rep
            and the count

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Two nodes are identified if their respective memberships are the same.
        Using this as an equivalence relation, the uids of the nodes are partitioned into
        equivalence classes. A single member of the equivalence class is chosen to represent
        the class followed by the number of members of the class.

        Example
        -------

            >>> h = Hypergraph(EntitySet('example',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])]))
            >>> h.incidence_dict
            {'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
            >>> h.collapse_nodes().incidence_dict
            {'E1': {frozenset({'a', 'b'})}, 'E2': {frozenset({'a', 'b'})}} ### Fix this
            >>> h.collapse_nodes(use_reps=True).incidence_dict
            {'E1': {('a', 2)}, 'E2': {('a', 2)}}

        """
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps ane return_counts are no longer supported keyword arguments and will throw
            an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        if self.nwhy:
            temp = self._collapse_nwhy(False, return_equivalence_classes)
            if return_equivalence_classes:
                return Hypergraph(temp[0], name, use_nwhy=self.nwhy), temp[1]
            else:
                return Hypergraph(temp, name, use_nwhy=self.nwhy)
        else:
            if self.isstatic:
                temp = self.dual().edges.collapse_identical_elements(
                    "_", return_equivalence_classes=return_equivalence_classes
                )
            else:
                temp = self.nodes.collapse_identical_elements(
                    "_", return_equivalence_classes=return_equivalence_classes
                )

            if return_equivalence_classes:
                return Hypergraph(temp[0], name, use_nwhy=self.nwhy).dual(), temp[1]
            else:
                return Hypergraph(temp, name, use_nwhy=self.nwhy).dual()

    def collapse_nodes_and_edges(
        self,
        name=None,
        use_reps=True,
        return_counts=True,
        return_equivalence_classes=False,
    ):
        """
        Returns a new hypergraph by collapsing nodes and edges.

        Parameters
        ----------

        name: str, optional, default: None

        use_reps: boolean, optional, default: False
            Choose a single element from the collapsed elements as a representative

        return_counts: boolean, optional, default: True
            if use_reps is True the new elements are keyed by a tuple of the rep
            and the count

        return_equivalence_classes: boolean, optional, default: False
            Returns a dictionary of edge equivalence classes keyed by frozen sets of nodes

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Collapses the Nodes and Edges EntitySets. Two nodes(edges) are duplicates
        if their respective memberships(elements) are the same. Using this as an
        equivalence relation, the uids of the nodes(edges) are partitioned into
        equivalence classes. A single member of the equivalence class is chosen to represent
        the class followed by the number of members of the class.

        Example
        -------

            >>> h = Hypergraph(EntitySet('example',elements=[Entity('E1', ['a','b']),Entity('E2',['a','b'])]))
            >>> h.incidence_dict
            {'E1': {'a', 'b'}, 'E2': {'a', 'b'}}
            >>> h.collapse_nodes_and_edges().incidence_dict   ### Fix this
            {('E1', 2): {('a', 2)}}

        """
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps ane return_counts are no longer supported keyword arguments and will throw
            an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        if return_equivalence_classes:
            temp, neq = self.collapse_nodes(
                name="temp", return_equivalence_classes=True
            )
            ntemp, eeq = temp.collapse_edges(name=name, return_equivalence_classes=True)
            return ntemp, neq, eeq
        else:
            temp = self.collapse_nodes(name="temp")
            return temp.collapse_edges(name=name)

    def restrict_to_edges(self, edgeset, name=None):
        """
        Constructs a hypergraph using a subset of the edges in hypergraph

        Parameters
        ----------
        edgeset: iterable of hashables or Entities
            A subset of elements of the hypergraph edges

        name: str, optional

        Returns
        -------
        new hypergraph : Hypergraph
        """
        if self._static:
            E = self._edges
            setsystem = E.restrict_to(sorted(E.indices(E.keys[0], list(edgeset))))
            return Hypergraph(setsystem, name=name, use_nwhy=self.nwhy)
        else:
            inneredges = set()
            for e in edgeset:
                if isinstance(e, Entity):
                    inneredges.add(e.uid)
                else:
                    inneredges.add(e)
            return Hypergraph({e: self.edges[e] for e in inneredges}, name=name)

    def restrict_to_nodes(self, nodeset, name=None):
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
        if self.isstatic:
            E = self.edges.restrict_to_levels((1, 0))
            setsystem = E.restrict_to(sorted(E.indices(E.keys[0], list(nodeset))))
            return Hypergraph(
                setsystem.restrict_to_levels((1, 0)), name=name, use_nwhy=self.nwhy
            )
        else:
            memberships = set()
            innernodes = set()
            for node in nodeset:
                # only use innernodes if they are part of the node set.
                if node in self.nodes:
                    innernodes.add(node)
                    memberships.update(set(self.nodes[node]._elements))
            newedgeset = dict()
            for e in memberships:
                if e in self.edges: # don't think this if-statement is necessary
                    temp = set(self.edges[e]._elements).intersection(innernodes)
                    # temp = set(self.edges[e]).issubset(innernodes) # in the induced sub-hypergraph case.
                    if temp:
                        newedgeset[e] = Entity(e, temp, **self.edges[e].properties)
            return Hypergraph(newedgeset, name=name)

    def toplexes(self, name=None, collapse=False, use_reps=False, return_counts=True):
        """
        Returns a :term:`simple hypergraph` corresponding to self.

        Warning
        -------
        Collapsing is no longer supported inside the toplexes method. Instead generate a new
        collapsed hypergraph and compute the toplexes of the new hypergraph.

        Parameters
        ----------
        name: str, optional, default: None

        # collapse: boolean, optional, default: False
        #     Should the hypergraph be collapsed? This would preserve a link between duplicate maximal sets.
        #     If False then only one of these sets will be used and uniqueness will be up to sets of equal size.

        # use_reps: boolean, optional, default: False
        #     If collapse=True then each toplex will be named by a representative of the set of
        #     equivalent edges, default is False (see collapse_edges).

        return_counts: boolean, optional, default: True
            # If collapse=True then each toplex will be named by a tuple of the representative
            # of the set of equivalent edges and their count

        """
        # TODO: There is a better way to do this....need to refactor
        if collapse:
            if len(self.edges) > 20:  # TODO: Determine how big is too big.
                warnings.warn(
                    "Collapsing a hypergraph can take a long time. It may be preferable to collapse the graph first and pickle it then apply the toplex method separately."
                )
            temp = self.collapse_edges()
        else:
            temp = self

        if collapse:
            msg = """
            collapse, return_counts, and use_reps are no longer supported keyword arguments 
            and will throw an error in the next release.
            """
            warnings.warn(msg, DeprecationWarning)

        thdict = dict()
        if self.nwhy:
            tops = self.g.toplexes()
            E = self.edges.restrict_to(tops)
            return Hypergraph(E, use_nwhy=True)
        else:
            tops = list()
            for e in temp.edges:
                flag = True
                old_tops = list(tops)
                for top in old_tops:
                    print("hi")
                    if set(temp.edges[e]).issubset(temp.edges[top]):
                        flag = False
                        break
                    elif set(temp.edges[top]).issubset(temp.edges[e]):
                        tops.remove(top)
                if flag:
                    tops += [e]
            return Hypergraph([temp.edges[t] for t in tops], name=name)

    def is_connected(self, s=1, edges=False):
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

        if self.isstatic:
            g = self.get_linegraph(s=s, edges=edges)
            if self.nwhy:
                return g.is_s_connected()
            else:
                return g.is_connected()
            return result
        else:
            if edges:
                A = self.edge_adjacency_matrix(s=s)
            else:
                A = self.adjacency_matrix(s=s)
            G = nx.from_scipy_sparse_matrix(A)
            return nx.is_connected(G)

    def singletons(self):
        """
        Returns a list of singleton edges. A singleton edge is an edge of
        size 1.

        Returns
        -------
        singles : list
            A list of edge uids.
        """
        if self.nwhy:
            return self.edges.translate(0, self.g.singletons())
        elif self.isstatic:
            M, rdict, cdict = self.incidence_matrix(index=True)
            idx = np.argmax(M.shape)  # which axis has fewest members? if 1 then columns
            cols = M.sum(idx)  # we add down the row index if there are fewer columns
            singles = list()
            for c in range(cols.shape[(idx + 1) % 2]):  # index along opposite axis
                if cols[idx * c, c * ((idx + 1) % 2)] == 1:
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
                            singles.append(cdict[c])
                    else:  # switch the role of r and c
                        r = np.argmax(M.getrow(c))
                        s = np.sum(M.getcol(r))
                        if s == 1:
                            singles.append(cdict[r])
            return singles
        else:
            return [id for id in self._edges if len(self._edges[id]) == 1]

    def remove_singletons(self, name=None):
        """XX
        Constructs clone of hypergraph with singleton edges removed.

        Parameters
        ----------
        name: str, optional, default: None

        Returns
        -------
        new hypergraph : Hypergraph

        """
        # self.remove_edges(self.singletons())
        E = [self.edges[e] for e in self.edges if e not in self.singletons()]
        return Hypergraph(E, name=name)

    def s_connected_components(self, s=1, edges=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-edge-connected components <s-edge-connected component>`
        or the :term:`s-node-connected components <s-connected component, s-node-connected component>`
        of the hypergraph.

        Parameters
        ----------
        s : int, optional, default: 1

        edges : boolean, optional, default: True
            If True will return edge components, if False will return node components
        return_singletons : bool, optional, default : False

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

        Yields
        ------
        s_connected_components : iterator
            Iterator returns sets of uids of the edges (or nodes) in the s-edge(node)
            components of hypergraph.

        """
        components = list()

        if self.nwhy:
            g = self.get_linegraph(s, edges=edges)
            if return_singletons:
                allobjects = set(self.edges) if edges == True else set(self.nodes)
                for c in g.s_connected_components():
                    comp = {self.get_name(nd, edges=edges) for nd in c}
                    allobjects.difference_update(comp)
                for c in g.s_connected_components():
                    yield {self.get_name(nd, edges=edges) for nd in c}
                for obj in allobjects:
                    yield {obj}
            else:
                for c in g.s_connected_components():
                    comp = {self.get_name(nd, edges=edges) for nd in c}
                    yield comp

        elif self.isstatic:
            g = self.get_linegraph(s, edges=edges)
            for c in nx.connected_components(g):
                if not return_singletons and len(c) == 1:
                    continue
                yield {self.get_name(n, edges=edges) for n in c}
        else:
            if edges:
                A, coldict = self.edge_adjacency_matrix(s=s, index=True)
                G = nx.from_scipy_sparse_matrix(A)
                # if not return_singletons:
                #     temp = [c for c in nx.connected_components(G) if len(c) > 1]
                # else:
                #     temp = nx.connected_components(G)
                for c in nx.connected_components(G):
                    if not return_singletons and len(c) == 1:
                        continue
                    yield {coldict[n] for n in c}
            else:
                A, rowdict = self.adjacency_matrix(s=s, index=True)
                G = nx.from_scipy_sparse_matrix(A)
                for c in nx.connected_components(G):
                    if not return_singletons:
                        if len(c) == 1:
                            continue
                    yield {rowdict[n] for n in c}

    def s_component_subgraphs(self, s=1, edges=True, return_singletons=False):
        """

        Returns a generator for the induced subgraphs of s_connected components.
        Removes singletons unless return_singletons is set to True. Computed using
        s-linegraph generated either by the hypergraph (edges=True) or its dual
        (edges = False)

        Parameters
        ----------
        s : int, optional, default: 1

        edges : boolean, optional, edges=False
            Determines if edge or node components are desired. Returns
            subgraphs equal to the hypergraph restricted to each set of nodes(edges) in the
            s-connected components or s-edge-connected components
        return_singletons : bool, optional

        Yields
        ------
        s_component_subgraphs : iterator
            Iterator returns subgraphs generated by the edges (or nodes) in the
            s-edge(node) components of hypergraph.

        """
        for idx, c in enumerate(
            self.s_components(s=s, edges=edges, return_singletons=return_singletons)
        ):
            if edges:
                yield self.restrict_to_edges(c, name=f"{self.name}:{idx}")
            else:
                yield self.restrict_to_nodes(c, name=f"{self.name}:{idx}")

    def s_components(self, s=1, edges=True, return_singletons=True):
        """
        Same as s_connected_components

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(
            s=s, edges=edges, return_singletons=return_singletons
        )

    def connected_components(self, edges=False, return_singletons=True):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(edges=edges, return_singletons=True)

    def connected_component_subgraphs(self, return_singletons=True):
        """
        Same as :meth:`s_component_subgraphs` with s=1. Returns iterator

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def components(self, edges=False, return_singletons=True):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(s=1, edges=edges)

    def component_subgraphs(self, return_singletons=False):
        """
        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(return_singletons=return_singletons)

    def node_diameters(self, s=1):
        """
        Returns the node diameters of the connected components in hypergraph.

        Parameters
        ----------
        list of the diameters of the s-components and
        list of the s-component nodes
        """
        if self.nwhy:
            g = self.get_linegraph(s, edges=False)
            if g.is_s_connected():
                return g.s_diameter()
            else:
                diameters = list()
                nodelists = list()
                for c in g.s_connected_components():
                    tc = self.edges.labs(1)[c]
                    nodelists.append(tc)
                    diameters.append(self.restrict_to_nodes(tc).node_diameters(s=s))
        else:
            A, coldict = self.adjacency_matrix(s=s, index=True)
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

    def edge_diameters(self, s=1):
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
        if self.nwhy:
            g = self.get_linegraph(s, edges=True)
            if g.is_s_connected():
                return g.s_diameter()
            else:
                diameters = list()
                edgelists = list()
                for c in g.s_connected_components():
                    tc = self.edges.labs(0)[c]
                    edgelists.append(tc)
                    diameters.append(self.restrict_to_edges(tc).edge_diameters(s=s))
        else:
            A, coldict = self.edge_adjacency_matrix(s=s, index=True)
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

    def diameter(self, s=1):
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
        if self.nwhy:
            g = self.get_linegraph(s, edges=False)
            if g.is_s_connected():
                return g.s_diameter()
            else:
                raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")
        else:
            A = self.adjacency_matrix(s=s)
            G = nx.from_scipy_sparse_matrix(A)
            if nx.is_connected(G):
                return nx.diameter(G)
            else:
                raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")

    def edge_diameter(self, s=1):
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
        if self.nwhy:
            g = self.get_linegraph(s, edges=True)
            if g.is_s_connected():
                return g.s_diameter()
            else:
                raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")
        else:
            A = self.edge_adjacency_matrix(s=s)
            G = nx.from_scipy_sparse_matrix(A)
            if nx.is_connected(G):
                return nx.diameter(G)
            else:
                raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")

    def distance(self, source, target, s=1):
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
        if self.isstatic:
            g = self.get_linegraph(s=s, edges=False)
            src = self.get_id(source, edges=False)
            tgt = self.get_id(target, edges=False)
            try:
                if self.nwhy:
                    d = g.s_distance(src, tgt)
                    if d == -1:
                        warnings.warn(f"No {s}-path between {source} and {target}")
                        return np.inf
                    else:
                        return d
                else:
                    return nx.shortest_path(g, src, tgt)
            except:
                warnings.warn(f"No {s}-path between {source} and {target}")
                return np.inf
        else:
            if isinstance(source, Entity):
                source = source.uid
            if isinstance(target, Entity):
                target = target.uid
            A, rowdict = self.adjacency_matrix(s=s, index=True)
            g = nx.from_scipy_sparse_matrix(A)
            rkey = {v: k for k, v in rowdict.items()}
            try:
                path = nx.shortest_path_length(g, rkey[source], rkey[target])
                return path
            except:
                warnings.warn(f"No {s}-path between {source} and {target}")
                return np.inf

    def edge_distance(self, source, target, s=1):
        """XX TODO: still need to return path and translate into user defined nodes and edges
        Returns the shortest s-walk distance between two edges in the hypergraph.

        Parameters
        ----------
        source : edge.uid or edge
            an edge in the hypergraph

        target : edge.uid or edge
            an edge in the hypergraph

        s : positive integer
            the number of intersections between pairwise consecutive edges

        TODO: add edge weights
        weight : None or string, optional, default: None
            if None then all edges have weight 1. If string then edge attribute
            string is used if available.


        Returns
        -------
        s- walk distance : the shortest s-walk edge distance
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
        if self.isstatic:
            g = self.get_linegraph(s=s, edges=True)
            src = self.get_id(source, edges=True)
            tgt = self.get_id(target, edges=True)
            try:
                if self.nwhy:
                    d = g.s_distance(src, tgt)
                    if d == -1:
                        warnings.warn(f"No {s}-path between {source} and {target}")
                        return np.inf
                    else:
                        return d
                else:
                    return nx.shortest_path(g, src, tgt)
            except:
                warnings.warn(f"No {s}-path between {source} and {target}")
                return np.inf
        else:
            if isinstance(source, Entity):
                source = source.uid
            if isinstance(target, Entity):
                target = target.uid
            A, coldict = self.edge_adjacency_matrix(s=s, index=True)
            g = nx.from_scipy_sparse_matrix(A)
            ckey = {v: k for k, v in coldict.items()}
            try:
                path = nx.shortest_path_length(g, ckey[source], ckey[target])
                return path
            except:
                warnings.warn(f"No {s}-path between {source} and {target}")
                return np.inf

    def dataframe(self, sort_rows=False, sort_columns=False):
        """
        Returns a pandas dataframe for hypergraph indexed by the nodes and
        with column headers given by the edge names.

        Parameters
        ----------
        sort_rows : bool, optional, default=True
            sort rows based on hashable node names
        sort_columns : bool, optional, default=True
            sort columns based on hashable edge names

        """

        mat, rdx, cdx = self.edges.incidence_matrix(index=True)
        index = [rdx[i] for i in rdx]
        columns = [cdx[j] for j in cdx]
        df = pd.DataFrame(mat.todense(), index=index, columns=columns)
        if sort_rows:
            df = df.sort_index()
        if sort_columns:
            df = df[sorted(columns)]
        return df

    @classmethod
    def from_bipartite(
        cls, B, set_names=("nodes", "edges"), name=None, static=False, use_nwhy=False
    ):
        """
        Static method creates a Hypergraph from a bipartite graph.

        Parameters
        ----------

        B: nx.Graph()
            A networkx bipartite graph. Each node in the graph has a property
            'bipartite' taking the value of 0 or 1 indicating a 2-coloring of the graph.

        set_names: iterable of length 2, optional, default = ['nodes','edges']
            Category names assigned to the graph nodes associated to each bipartite set

        name: hashable

        static: bool

        Returns
        -------
         : Hypergraph

        Notes
        -----
        A partition for the nodes in a bipartite graph generates a hypergraph.

            >>> import networkx as nx
            >>> B = nx.Graph()
            >>> B.add_nodes_from([1, 2, 3, 4], bipartite=0)
            >>> B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
            >>> B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
            >>> H = Hypergraph.from_bipartite(B)
            >>> H.nodes, H.edges
            # output: (EntitySet(_:Nodes,[1, 2, 3, 4],{}), EntitySet(_:Edges,['b', 'c', 'a'],{}))

        """
        msg = "from_bipartite() is now a method in the read_write module"
        warnings.warn(msg, DeprecationWarning)
        edges = []
        nodes = []
        for n, d in B.nodes(data=True):
            if d["bipartite"] == 0:
                nodes.append(n)
            else:
                edges.append(n)

        if not bipartite.is_bipartite_node_set(B, nodes):
            raise HyperNetXError("Error: Method requires a 2-coloring of a bipartite graph.")
        
        if static:
            elist = []
            for e in list(B.edges):
                if e[0] in nodes:
                    elist.append([e[0], e[1]])
                else:
                    elist.append([e[1], e[0]])
            df = pd.DataFrame(elist, columns=set_names)
            E = StaticEntitySet(entity=df)
            return Hypergraph(E, name=name, use_nwhy=use_nwhy)
        else:
            edge_dict = {e: list(B.neighbors(e)) for e in edges}
            return Hypergraph(edge_dict, name=name)


    @classmethod
    def from_numpy_array(
        cls,
        M,
        node_names=None,
        edge_names=None,
        node_label="nodes",
        edge_label="edges",
        name=None,
        key=None,
        static=False,
        use_nwhy=False,
    ):
        """
        Create a hypergraph from a real valued matrix represented as a 2 dimensionsl numpy array.
        The matrix is converted to a matrix of 0's and 1's so that any truthy cells are converted to 1's and
        all others to 0's.

        Parameters
        ----------
        M : real valued array-like object, 2 dimensions
            representing a real valued matrix with rows corresponding to nodes and columns to edges

        node_names : object, array-like, default=None
            List of node names must be the same length as M.shape[0].
            If None then the node names correspond to row indices with 'v' prepended.

        edge_names : object, array-like, default=None
            List of edge names must have the same length as M.shape[1].
            If None then the edge names correspond to column indices with 'e' prepended.

        name : hashable

        key : (optional) function
            boolean function to be evaluated on each cell of the array,
            must be applicable to numpy.array

        Returns
        -------
         : Hypergraph

        Note
        ----
        The constructor does not generate empty edges.
        All zero columns in M are removed and the names corresponding to these
        edges are discarded.


        """
        msg = "from_bipartite() is now a method in the read_write module"
        warnings.warn(msg, DeprecationWarning)
        # Create names for nodes and edges
        # Validate the size of the node and edge arrays

        M = np.array(M)
        if len(M.shape) != (2):
            raise HyperNetXError("Input requires a 2 dimensional numpy array")
        # apply boolean key if available
        if key:
            M = key(M)

        if node_names is not None:
            nodenames = np.array(node_names)
            if len(nodenames) != M.shape[0]:
                raise HyperNetXError(
                    "Number of node names does not match number of rows."
                )
        else:
            nodenames = np.array([f"v{idx}" for idx in range(M.shape[0])])

        if edge_names is not None:
            edgenames = np.array(edge_names)
            if len(edgenames) != M.shape[1]:
                raise HyperNetXError(
                    "Number of edge_names does not match number of columns."
                )
        else:
            edgenames = np.array([f"e{jdx}" for jdx in range(M.shape[1])])

        if static or use_nwhy:
            arr = np.array(M)
            if key:
                arr = key(arr) * 1
            arr = arr.transpose()
            labels = OrderedDict([(edge_label, edgenames), (node_label, nodenames)])
            E = StaticEntitySet(arr=arr, labels=labels)
            return Hypergraph(E, name=name, use_nwhy=use_nwhy)

        else:
            # Remove empty column indices from M columns and edgenames
            colidx = np.array([jdx for jdx in range(M.shape[1]) if any(M[:, jdx])])
            colidxsum = np.sum(colidx)
            if not colidxsum:
                return Hypergraph()
            else:
                M = M[:, colidx]
                edgenames = edgenames[colidx]
                edict = dict()
                # Create an EntitySet of edges from M
                for jdx, e in enumerate(edgenames):
                    edict[e] = nodenames[
                        [idx for idx in range(M.shape[0]) if M[idx, jdx]]
                    ]
                return Hypergraph(edict, name=name)

    @classmethod
    def from_dataframe(
        cls,
        df,
        columns=None,
        rows=None,
        name=None,
        fillna=0,
        transpose=False,
        transforms=[],
        key=None,
        node_label="nodes",
        edge_label="edges",
        static=False,
        use_nwhy=False,
    ):
        """
        Create a hypergraph from a Pandas Dataframe object using index to label vertices
        and Columns to label edges. The values of the dataframe are transformed into an 
        incidence matrix.  
        Note this is different than passing a dataframe directly
        into the Hypergraph constructor. The latter automatically generates a static hypergraph
        with edge and node labels given by the cell values.

        Parameters
        ----------
        df : Pandas.Dataframe
            a real valued dataframe with a single index

        columns : (optional) list, default = None
            restricts df to the columns with headers in this list.

        rows : (optional) list, default = None
            restricts df to the rows indexed by the elements in this list.

        name : (optional) string, default = None

        fillna : float, default = 0
            a real value to place in empty cell, all-zero columns will not generate
            an edge.

        transpose : (optional) bool, default = False
            option to transpose the dataframe, in this case df.Index will label the edges
            and df.columns will label the nodes, transpose is applied before transforms and
            key

        transforms : (optional) list, default = []
            optional list of transformations to apply to each column,
            of the dataframe using pd.DataFrame.apply().
            Transformations are applied in the order they are
            given (ex. abs). To apply transforms to rows or for additional
            functionality, consider transforming df using pandas.DataFrame methods
            prior to generating the hypergraph.

        key : (optional) function, default = None
            boolean function to be applied to dataframe. Must be defined on numpy
            arrays.

        See also
        --------
        from_numpy_array())


        Returns
        -------
        : Hypergraph

        Notes
        -----
        The `from_dataframe` constructor does not generate empty edges.
        All-zero columns in df are removed and the names corresponding to these
        edges are discarded.
        Restrictions and data processing will occur in this order:

            1. column and row restrictions
            2. fillna replace NaNs in dataframe
            3. transpose the dataframe
            4. transforms in the order listed
            5. boolean key

        This method offers the above options for wrangling a dataframe into an incidence
        matrix for a hypergraph. For more flexibility we recommend you use the Pandas
        library to format the values of your dataframe before submitting it to this
        constructor.

        """

        if type(df) != pd.core.frame.DataFrame:
            raise HyperNetXError("Error: Input object must be a pandas dataframe.")

        if columns:
            df = df[columns]
        if rows:
            df = df.loc[rows]

        df = df.fillna(fillna)
        if transpose:
            df = df.transpose()

        # node_names = np.array(df.index)
        # edge_names = np.array(df.columns)

        for t in transforms:
            df = df.apply(t)
        if key:
            mat = key(df.values) * 1
        else:
            mat = df.values * 1

        params = {
            "node_names": np.array(df.index),
            "edge_names": np.array(df.columns),
            "name": name,
            "node_label": node_label,
            "edge_label": edge_label,
            "static": static,
            "use_nwhy": use_nwhy,
        }
        return cls.from_numpy_array(mat, **params)


# end of Hypergraph class


def _make_3_arrays(mat):
    arr = coo_matrix(mat)
    return arr.row, arr.col, arr.data
