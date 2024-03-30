# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.
from __future__ import annotations

import warnings

warnings.filterwarnings("default", category=DeprecationWarning)

from copy import deepcopy
from collections import defaultdict
from collections.abc import Sequence, Iterable
from typing import Optional, Any, TypeVar, Union, Mapping, Hashable

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix, csr_matrix

from hypernetx.exception import HyperNetXError
from hypernetx.classes.factory import (
    dataframe_factory_method,
    dict_factory_method,
    list_factory_method,
)
from hypernetx.classes.incidence_store import IncidenceStore
from hypernetx.classes.property_store import PropertyStore
from hypernetx.classes.hyp_view import HypergraphView

__all__ = ["Hypergraph"]

T = TypeVar("T", bound=Union[str, int])


class Hypergraph:

    def __init__(
        self,
        ### these are for the incidence pairs and their properties
        ### format for properties must follow from incidence pairs
        ### so that properties are provided either in the dataframe
        ### or as part of a nested dictionary.
        setsystem: Optional[
            pd.DataFrame
            | np.ndarray
            | Mapping[T, Iterable[T]]
            | Iterable[Iterable[T]]
            | Mapping[T, Mapping[T, Mapping[str, Any]]]
        ] = None,
        default_cell_weight: float | int = 1,  ### we will no longer support a sequence
        edge_col: str | int = 0,
        node_col: str | int = 1,
        cell_weight_col: Optional[str | int] = "weight",
        cell_properties: Optional[
            Sequence[str | int] | Mapping[T, Mapping[T, Mapping[str, Any]]]
        ] = None,
        misc_cell_properties_col: Optional[str | int] = None,
        aggregate_by: str | dict[str, str] = "first",
        ### Format for properties can be either a dataframe indexed on uid
        ### or with first column equal to uid or a dictionary
        ### use these for a single properties list
        properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        prop_uid_col: (
            str | int | None
        ) = None,  ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_properties_col: Optional[str | int] = None,
        weight_prop_col: str | int = "weight",
        default_weight: float = 1.0,
        ### these are just for properties on the edges - ignored if properties exists
        edge_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        edge_uid_col: (
            str | int | None
        ) = None,  ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_edge_properties_col: Optional[str | int] = None,
        edge_weight_prop_col: str | int = "weight",
        default_edge_weight: float | int = 1,
        ### these are just for properties on the nodes - ignored if properties exists
        node_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        node_uid_col: (
            str | int | None
        ) = None,  ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_node_properties_col: Optional[str | int] = None,
        node_weight_prop_col: str | int = "weight",
        default_node_weight: float | int = 1,
        name: Optional[str] = None,
        **kwargs,
    ):

        #### Use a Factory Method to create 4 stores
        ## df = Incidence Store from structural data
        ## edges,nodes,incidences = Property stores tied to ids in df
        ## Incidences - uses df, dictionary - assigns keys to incidence
        ## pairs and/or removes duplicates,
        ## identifies keys for edges and nodes

        ## Construct 3 HypergraphViews to tie these together?
        ## Incidences - links keys from 3 stores, and incidences with properties
        ## Edges - reconciles keys from Incidences against a Property Store
        ## Nodes - reconciles keys from Incidences against a Property Store

        # decision tree to route data by type OR user specified type
        # to correct factory method for each of 3 property stores
        # and 1 incidence store
        # 3 calls to the different factory methods (data => df for Property Store)
        # 1 call to the incidence factory method (setsystem => df for Incidence Store)

        type_dict = {
            "DataFrame": dataframe_factory_method,
            "dict": dict_factory_method,
            "OrderedDict": dict_factory_method,
            "defaultdict": dict_factory_method,
            "list": list_factory_method,
        }

        ## dataframe_factory_method(setsystem_df,uid_cols=[edge_col,node_col],weight_col,default_weight,misc_properties,aggregate_by)
        ## dataframe_factory_method(edge_properties_df,uid_cols=[edge_uid_col],edge_weight_col,default_edge_weight,misc_edge_properties)

        if setsystem is None:
            setsystem = pd.DataFrame(
                columns=["edges", "nodes", "weight", "misc_properties"]
            )
        setsystem_type = type(setsystem).__name__
        if setsystem_type in type_dict:
            df = type_dict[setsystem_type](
                setsystem,
                2,
                uid_cols=[edge_col, node_col],
                weight_col=cell_weight_col,
                default_weight=default_cell_weight,
                misc_properties_col=misc_cell_properties_col,
                aggregate_by=aggregate_by,
            )
            ## dataframe_factory_method(edf,uid_cols=[uid_col],weight_col,default_weight,misc_properties)
            ## multi index set by uid_cols = [edge_col,node_col]
            incidence_store = IncidenceStore(
                pd.DataFrame(list(df.index), columns=["edges", "nodes"])
            )
            incidence_propertystore = PropertyStore(
                data=df, default_weight=default_cell_weight
            )
            self._E = HypergraphView(incidence_store, 2, incidence_propertystore)
            ## if no properties PropertyStore should store in the most efficient way
        else:
            raise HyperNetXError("setsystem data type not supported")
        ## check if there is another constructor they could use.

        if properties is not None:
            property_type = type(properties).__name__
            if property_type in type_dict:
                dfp = type_dict[property_type](
                    properties,
                    0,
                    weight_col=weight_prop_col,
                    default_weight=default_weight,
                    misc_properties_col=misc_properties_col,
                )
                all_propertystore = PropertyStore(
                    data=dfp, default_weight=default_weight
                )
                self._edges = HypergraphView(incidence_store, 0, all_propertystore)
                self._nodes = HypergraphView(incidence_store, 1, all_propertystore)
        else:
            if edge_properties is not None:
                edge_property_type = type(edge_properties).__name__
                if edge_property_type in type_dict:
                    edfp = type_dict[edge_property_type](
                        edge_properties,
                        0,
                        weight_col=edge_weight_prop_col,
                        default_weight=default_edge_weight,
                        misc_properties_col=misc_edge_properties_col,
                    )
                    edge_propertystore = PropertyStore(
                        edfp, default_weight=default_edge_weight
                    )
                else:
                    edge_properties = PropertyStore(default_weight=default_edge_weight)
            else:
                edge_propertystore = PropertyStore(default_weight=default_edge_weight)
            self._edges = HypergraphView(incidence_store, 0, edge_propertystore)

            if node_properties is not None:
                node_property_type = type(node_properties).__name__
                if node_property_type in type_dict:
                    ndfp = type_dict[node_property_type](
                        node_properties,
                        1,
                        weight_col=node_weight_prop_col,
                        default_weight=default_node_weight,
                        misc_properties_col=misc_node_properties_col,
                    )
                    node_propertystore = PropertyStore(
                        ndfp, default_weight=default_node_weight
                    )
                else:
                    node_propertystore = PropertyStore(
                        default_weight=default_node_weight
                    )
            else:
                node_propertystore = PropertyStore(default_weight=default_node_weight)
            self._nodes = HypergraphView(incidence_store, 1, node_propertystore)

        self._dataframe = self.dataframe
        self._set_default_state()
        self.name = name
        self.__dict__.update(locals())

    @property
    def edges(self):
        """
        Object associated with edges.

        Returns
        -------
        EntitySet
        """
        return self._edges

    @property
    def nodes(self):
        """
        Object associated with nodes.

        Returns
        -------
        EntitySet
        """
        return self._nodes

    @property
    def incidences(self):
        """
        Object associated with incidence pairs

        Returns
        -------
        _type_
            _description_
        """
        return self._E

    @property
    def dataframe(self):
        """Returns dataframe of incidence properties
        as dataframe with edges and nodes in columns.

        Returns
        -------
        pd.DataFrame
        """
        df = self._E.dataframe.reset_index()
        return df

    @property
    def properties(self):
        """Returns incidence properties

        Returns
        -------
        pd.DataFrame or Dictionary?
        """
        return self._E.properties

    def incidence_matrix(self, index=False, weights=False):
        """
        _summary_

        Parameters
        ----------
        index : bool, optional
            _description_, by default False
        use_weights : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        e, n = self._state_dict["data"].T
        if weights == True:
            data = self._E.dataframe['weight']
        else:
            data = np.ones(len(e)).astype(int)
        mat = csr_matrix((data, (n, e)))
        if index == False:
            return mat
        else:
            return (
                mat,
                self._state_dict["labels"]["nodes"],
                self._state_dict["labels"]["edges"],
            )

    def incidence_dataframe(self, weights=False):
        mat, rindex, cindex = self.incidence_matrix(index=True, weights=weights)
        return pd.DataFrame(mat.toarray(), columns=cindex, index=rindex)

    @property
    def edge_props(self):
        """Dataframe of edge properties
        indexed on edge ids

        Returns
        -------
        pd.DataFrame or Dictionary?
        """
        return self._edges.dataframe

    @property
    def node_props(self):
        """Dataframe of node properties
        indexed on node ids

        Returns
        -------
        pd.DataFrame or Dictionary?
        """
        return self._nodes.dataframe

    @property
    def incidence_dict(self):
        """
        Dictionary keyed by edge uids with values the uids of nodes in each
        edge

        Returns
        -------
        dict

        """
        return self._E.elements

    @property
    def shape(self):
        """
        (number of nodes, number of edges)

        Returns
        -------
        tuple

        """
        return len(self._nodes), len(self._edges)  ## incidence store call?

    def __str__(self):
        """
        String representation of hypergraph

        Returns
        -------
        str

        """
        return f"{self.name}, <class 'hypernetx.classes.hypergraph.Hypergraph'>"

    def __repr__(self):
        """
        String representation of hypergraph

        Returns
        -------
        str

        """
        return f"{self.name}, hypernetx.classes.hypergraph.Hypergraph"

    def __len__(self):
        """
        Number of nodes

        Returns
        -------
        int

        """
        return len(self._nodes)

    def __iter__(self):
        """
        Iterate over the nodes of the hypergraph

        Returns
        -------
        dict_keyiterator

        """
        return iter(self._nodes)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.nodes

        Parameters
        ----------
        item : hashable or EntitySet

        """
        return item in self._nodes

    def __getitem__(self, node):  ## TODO: Do we change all "cell" refs to "incidence"?
        """
        Returns the neighbors of node

        Parameters
        ----------
        node : EntitySet or hashable
            If hashable, then must be uid of node in hypergraph

        Returns
        -------
        neighbors(node) : iterator

        """
        return self.neighbors(node)

    def get_cell_properties(
        self, edge: str, node: str, prop_name: Optional[str] = None
    ) -> Any | dict[str, Any]:
        """Get cell properties on a specified edge and node

        Parameters
        ----------
        edge : str
            edgeid
        node : str
            nodeid
        key : int, optional,
            incidence store id, by default None
        prop_name : str, optional
            name of a cell property; if None, all cell properties will be returned

        Returns
        -------
        : int or str or dict of {str: any}
            cell property value if `prop_name` is provided, otherwise ``dict`` of all
            cell properties and values
        """
        return self._E.properties(
            (edge, node), prop_name=prop_name
        )  ### get_property from hyp_view

    def get_properties(self, uid, level=0, prop_name=None):
        """Returns an object's specific property or all properties
        ### Change to level is 0,1,2 and call props from correct store
        Parameters
        ----------
        uid : hashable
            edge or node id
        level : int | None , optional, default = None
            if separate edge and node properties then enter 0 for edges
            and 1 for nodes.
        prop_name : str | None, optional, default = None
            if None then all properties associated with the object will  be
            returned.

        Returns
        -------
        : str or dict
            single property or dictionary of properties
        """

        if level == 0:
            store = self._edges
        elif level == 1:
            store = self._nodes
        elif level == 2:
            store = self._E
        if prop_name is None:  ## rewrite for edges and nodes.
            return store[uid].properties
        else:
            return store[uid].__getattr__(prop_name, None)

    def get_linegraph(self, s=1, edges=True):
        """
        Creates an ::term::s-linegraph for the Hypergraph.
        If edges=True (default)then the edges will be the vertices of the line
        graph. Two vertices are connected by an s-line-graph edge if the
        corresponding hypergraph edges intersect in at least s hypergraph nodes.
        If edges=False, the hypergraph nodes will be the vertices of the line
        graph. Two vertices are connected if the nodes they correspond to share
        at least s incident hyper edges.

        Parameters
        ----------
        s : int
            The width of the connections.
        edges : bool, optional, default = True
            Determine if edges or nodes will be the vertices in the linegraph.

        Returns
        -------
        nx.Graph
            A NetworkX graph.
        """
        d = self._state_dict
        key = "sedgelg" if edges else "snodelg"
        if s in d[key]:
            return d[key][s]

        if edges:  ### Amaplist needs a dictionary returned for properties.
            A, Amap = self.edge_adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self._edges[k].properties) for k in Amap]
        else:
            A, Amap = self.adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self._nodes[k].properties) for k in Amap]

        ### TODO: add key function to compute weights lambda x,y : funcval

        A = np.array(np.nonzero(A))
        e1 = np.array([Amap[idx] for idx in A[0]])
        e2 = np.array([Amap[idx] for idx in A[1]])
        A = np.array([e1, e2]).T
        g = nx.Graph()
        g.add_edges_from(A)
        g.add_nodes_from(Amaplst)
        d[key][s] = g
        return g

    def set_state(self, **kwargs):
        """
        Allow state_dict updates from outside of class. Use with caution.

        Parameters
        ----------
        **kwargs
            key=value pairs to save in state dictionary
        """
        self._state_dict.update(kwargs)

    def _set_default_state(self, empty=False):
        """Populate state_dict with default values
        This may change with HypegraphView since data is no
        longer required to be in a specific structure
        """
        self._state_dict = {}
        df = self._E.incidence_store.data
        self._state_dict["dataframe"] = df

        if empty:
            self._state_dict["labels"] = {"edges": np.array([]), "nodes": np.array([])}
            self._state_dict["data"] = np.array([[], []])
        else:
            df.edges = df.edges.astype("category")
            df.nodes = df.nodes.astype("category")
            self._state_dict["labels"] = {
                "edges": np.array(df["edges"].cat.categories),
                "nodes": np.array(df["nodes"].cat.categories),
            }
            self._state_dict["data"] = np.array(
                [df["edges"].cat.codes, df["nodes"].cat.codes], dtype=int
            ).T

        self._state_dict["snodelg"] = dict()  ### s: nx.graph
        self._state_dict["sedgelg"] = dict()
        self._state_dict["neighbors"] = defaultdict(dict)  ### s: {node: neighbors}
        self._state_dict["edge_neighbors"] = defaultdict(
            dict
        )  ### s: {edge: edge_neighbors}
        self._state_dict["adjacency_matrix"] = dict()  ### s: scipy.sparse.csr_matrix
        self._state_dict["edge_adjacency_matrix"] = dict()

    def edge_size_dist(self):
        """
        Returns the size for each edge

        Returns
        -------
        np.array

        """

        if "edge_size_dist" not in self._state_dict:
            dist = np.array(np.sum(self.incidence_matrix(), axis=0))[0].tolist()
            self.set_state(edge_size_dist=dist)
            return dist
        else:
            return self._state_dict["edge_size_dist"]

    def degree(self, node, s=1, max_size=None):
        """
        The number of edges of size at least s and at most
        max_size that contain the node.

        Parameters
        ----------
        node : hashable
            identifier for the node.
        s : positive integer, optional, default 1
            smallest size of edge to consider in degree
        max_size : positive integer or None, optional, default = None
            largest size of edge to consider in degree

        Returns
        -------
         : int

        """
        if s == 1 and max_size == None:
            return len(self._nodes.memberships[node])
        else:
            memberships = set()
            for edge in self._nodes.memberships[node]:
                size = len(self.edges[edge])
                if size >= s and (max_size is None or size <= max_size):
                    memberships.add(edge)
            ### This could possibly be done more efficiently on the dataframe.
            return len(memberships)

    def size(self, edge, nodeset=None):
        """
        The number of nodes in nodeset that belong to edge.
        If nodeset is None then returns the size of edge

        Parameters
        ----------
        edge : hashable
            The uid of an edge in the hypergraph

        Returns
        -------
        size : int

        """
        if nodeset is not None:
            return len(set(nodeset).intersection(set(self.edges[edge])))

        return len(self.edges[edge])

    def number_of_nodes(self, nodeset=None):  ## TODO: Not sure if needed
        """
        The number of nodes in nodeset belonging to hypergraph.

        Parameters
        ----------
        nodeset : an interable of Entities, optional, default = None
            If None, then return the number of nodes in hypergraph.

        Returns
        -------
        number_of_nodes : int

        """
        if nodeset is not None:
            return len([n for n in self.nodes if n in nodeset])

        return len(self.nodes)

    def number_of_edges(self, edgeset=None):  ## TODO: Not sure what this was for
        """
        The number of edges in edgeset belonging to hypergraph.

        Parameters
        ----------
        edgeset : an iterable of Entities, optional, default = None
            If None, then return the number of edges in hypergraph.

        Returns
        -------
        number_of_edges : int
        """
        if edgeset:
            return len([e for e in self.edges if e in edgeset])

        return len(self.edges)

    def order(self):
        """
        The number of nodes in hypergraph.

        Returns
        -------
        order : int
        """
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
        node : hashable or EntitySet
            uid for a node in hypergraph or the node Entity

        s : int, list, optional, default = 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        neighbors : list
            s-neighbors share at least s edges in the hypergraph

        """
        if node not in self.nodes:
            print(f"{node} is not in hypergraph {self.name}.")
            return None
        if node in self._state_dict["neighbors"][s]:
            return self._state_dict["neighbors"][s][node]
        else:
            M = self.incidence_matrix()
            rdx = self._state_dict["labels"]["nodes"]
            jdx = np.where(rdx == node)
            idx = (M[jdx].dot(M.T) >= s) * 1
            idx = np.nonzero(idx)[1]
            neighbors = list(rdx[idx])
            if len(neighbors) > 0:
                neighbors.remove(node)
                self._state_dict["neighbors"][s][node] = neighbors
            else:
                self._state_dict["neighbors"][s][node] = []
        return neighbors

    def edge_neighbors(self, edge, s=1):
        """
        The edges in hypergraph which share s nodes(s) with edge.

        Parameters
        ----------
        edge : hashable or EntitySet
            uid for a edge in hypergraph or the edge Entity

        s : int, list, optional, default = 1
            Minimum number of nodes shared by neighbors edge node.

        Returns
        -------
         : list
            List of edge neighbors

        """

        if edge not in self.edges:
            print(f"Edge is not in hypergraph {self.name}.")
            return None
        if edge in self._state_dict["edge_neighbors"][s]:
            return self._state_dict["edge_neighbors"][s][edge]
        else:
            M = self.incidence_matrix()
            cdx = self._state_dict["labels"]["edges"]
            jdx = np.where(cdx == edge)
            idx = (M.T[jdx].dot(M) >= s) * 1
            idx = np.nonzero(idx)[1]
            edge_neighbors = list(cdx[idx])
            if len(edge_neighbors) > 0:
                edge_neighbors.remove(edge)
                self._state_dict["edge_neighbors"][s][edge] = edge_neighbors
            else:
                self._state_dict["edge_neighbors"][s][edge] = []
            return edge_neighbors

    def adjacency_matrix(self, s=1, index=False, remove_empty_rows=False):
        """
        The :term:`s-adjacency matrix` for the hypergraph.

        Parameters
        ----------
        s : int, optional, default = 1

        index: boolean, optional, default = False
            if True, will return the index of ids for rows and columns

        remove_empty_rows: boolean, optional, default = False

        Returns
        -------
        adjacency_matrix : scipy.sparse.csr.csr_matrix

        node_index : list
            index of ids for rows and columns

        """
        try:
            A = self._state_dict["adjacency_matrix"][s]
        except:
            M = self.incidence_matrix()
            A = M @ (M.T)
            A.setdiag(0)
            A = (A >= s) * 1
            self._state_dict["adjacency_matrix"][s] = A
        if index == True:
            return A, self._state_dict["labels"]["nodes"]
        else:
            return A

    def edge_adjacency_matrix(self, s=1, index=False):
        """
        The :term:`s-adjacency matrix` for the dual hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        index: boolean, optional, default = False
            if True, will return the index of ids for rows and columns

        Returns
        -------
        edge_adjacency_matrix : scipy.sparse.csr.csr_matrix

        edge_index : list
            index of ids for rows and columns

        Notes
        -----
        This is also the adjacency matrix for the line graph.
        Two edges are s-adjacent if they share at least s nodes.
        If remove_zeros is True will return the auxillary matrix

        """
        try:
            A = self._state_dict["edge_adjacency_matrix"][s]
        except:
            M = self.incidence_matrix()
            A = (M.T) @ (M)
            A.setdiag(0)
            A = (A >= s) * 1
            self._state_dict["edge_adjacency_matrix"][s] = A
        if index == True:
            return A, self._state_dict["labels"]["edges"]
        else:
            return A

    def auxiliary_matrix(self, s=1, node=True, index=False):
        """
        The unweighted :term:`s-edge or node auxiliary matrix` for hypergraph

        Parameters
        ----------
        s : int, optional, default = 1
        node : bool, optional, default = True
            whether to return based on node or edge adjacencies

        Returns
        -------
        auxiliary_matrix : scipy.sparse.csr.csr_matrix
            Node/Edge adjacency matrix with empty rows and columns
            removed
        index : np.array
            row and column index of userids

        """
        if node == True:
            A, Amap = self.adjacency_matrix(s, index=True)
        else:
            A, Amap = self.edge_adjacency_matrix(s, index=True)

        idx = np.nonzero(np.sum(A, axis=1))[0]
        if len(idx) < A.shape[0]:
            B = A[idx][:, idx]
        else:
            B = A
        if index:
            return B, Amap[idx]
        else:
            return B

    def bipartite(self, keep_data=False, directed=False):  ## TODO share data with graph
        """
        Constructs the networkX bipartite graph associated to hypergraph.

        Returns
        -------
        bipartite : nx.Graph()

        Notes
        -----
        Creates a bipartite networkx graph from hypergraph.
        The nodes and (hyper)edges of hypergraph become the nodes of bipartite
        graph. For every (hyper)edge e in the hypergraph and node n in e there
        is an edge (n,e) in the graph.

        """
        if directed == True:
            B = nx.DiGraph()
        else:
            B = nx.Graph()
        if keep_data == False:
            B.add_nodes_from(self.edges, bipartite=0)
            B.add_nodes_from(self.nodes, bipartite=1)
            B.add_edges_from(
                [
                    (e, v)
                    for v in self._nodes.memberships
                    for e in self._nodes.memberships[v]
                ]
            )
        else:
            for nd in self.nodes:
                B.add_node(nd, bipartite=1, **self.nodes[nd].properties)
            for ed in self.edges:
                B.add_node(ed, bipartite=0, **self.edges[ed].properties)
            B.add_edges_from([(*d, self._E[d].properties) for d in self._E])
        return B

    def _construct_hyp_from_stores(
        self, incidence_df, edge_ps=None, node_ps=None, name=None
    ):

        h = Hypergraph()

        incidence_store = IncidenceStore(
            pd.DataFrame(incidence_df.index.tolist(), columns=["edges", "nodes"])
        )
        incidence_ps = PropertyStore(incidence_df)
        h._E = HypergraphView(incidence_store, 2, incidence_ps)

        if edge_ps is None:
            h._edges = HypergraphView(incidence_store, 0, self._edges.property_store)
        else:
            h._edges = HypergraphView(incidence_store, 0, edge_ps)

        if node_ps is None:
            h._nodes = HypergraphView(incidence_store, 1, self._nodes.property_store)
        else:
            h._nodes = HypergraphView(incidence_store, 1, node_ps)
        h._set_default_state()
        h.name = name
        h._dataframe = h.dataframe
        return h

    def dual(self, name=None, share_properties=True):
        """
        Constructs a new hypergraph with roles of edges and nodes of hypergraph
        reversed.

        Parameters
        ----------
        name : hashable, optional

        switch_names : bool, optional, default = True
            reverses edge_col and node_col names
            unless edge_col = 'edges' and node_col = 'nodes'

        Returns
        -------
        : hypergraph

        """

        C = self.dataframe.columns.tolist()

        dsetsystem = (
            self.dataframe[[C[1], C[0]] + C[2:]]
            .rename(columns={"edges": "nodes", "nodes": "edges"})
            .set_index(["edges", "nodes"])
        )

        if share_properties == False:
            edge_ps = PropertyStore(self._nodes.dataframe.copy(deep=True))
            node_ps = PropertyStore(self._edges.dataframe.copy(deep=True))
        else:
            edge_ps = self._nodes
            node_ps = self._edges

        hdual = self._construct_hyp_from_stores(
            dsetsystem,
            edge_ps=edge_ps,
            node_ps=node_ps,
            name=name or str(self.name) + "_dual",
        )
        return hdual

    ###### Collapse methods now handled completely in the incidence store
    def equivalence_classes(self, edges=True):
        level = 0 if edges == True else 1
        return self._E.incidence_store.equivalence_classes(level=level)

    def collapse_edges(
        self,
        name=None,
        use_uids=None,
        return_equivalence_classes=False,
        share_properties=False,
    ):
        """
        Constructs a new hypergraph gotten by identifying edges containing the
        same nodes

        Parameters
        ----------
        name : hashable, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes

        Returns
        -------
        new hypergraph : Hypergraph
            Equivalent edges are collapsed to a single edge named by a
            representative of the equivalent edges followed by a colon and the
            number of edges it represents.

        equivalence_classes : dict
            A dictionary keyed by representative edge names with values equal
            to the edges in its equivalence class

        Notes
        -----
        Two edges are identified if their respective elements are the same.
        Using this as an equivalence relation, the uids of the edges are
        partitioned into equivalence classes.

        A single edge from the collapsed edges followed by a colon and the
        number of elements in its equivalence class as uid for the new edge


        """
        df, eclasses = self._E.incidence_store.collapse_identical_elements(
            0, use_keys=use_uids, return_equivalence_classes=True
        )
        incidence_pairs = [tuple(d) for d in df.values]
        incidence_df = self._E.properties.loc[incidence_pairs]
        ekeys = list(eclasses.keys())
        ekeys = list(set(ekeys).intersection(self.edges.properties.index))
        if share_properties == True:
            H = _construct_hyp_from_stores(incidence_df)
        else:
            edge_ps = PropertyStore(self.edges.properties.copy(deep=True).loc[ekeys])
            node_ps = PropertyStore(self.nodes.properties.copy(deep=True))
            H = self._construct_hyp_from_stores(
                incidence_df, edge_ps=edge_ps, node_ps=node_ps, name=name
            )
        if return_equivalence_classes == True:
            return H, eclasses
        else:
            return H

    def collapse_nodes(
        self,
        name=None,
        use_uids=None,
        return_equivalence_classes=False,
        share_properties=False,
    ):
        """
        Constructs a new hypergraph gotten by identifying nodes contained in the
        same edges

        Parameters
        ----------
        name : hashable, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of node equivalence classes

        Returns
        -------
        new hypergraph : Hypergraph
            Equivalent edges are collapsed to a single edge named by a
            representative of the equivalent edges followed by a colon and the
            number of edges it represents.

        equivalence_classes : dict
            A dictionary keyed by representative edge names with values equal
            to the edges in its equivalence class

        Notes
        -----
        Two edges are identified if their respective elements are the same.
        Using this as an equivalence relation, the uids of the edges are
        partitioned into equivalence classes.

        A single edge from the collapsed edges followed by a colon and the
        number of elements in its equivalence class as uid for the new edge


        """
        df, eclasses = self._E.incidence_store.collapse_identical_elements(
            1, use_keys=use_uids, return_equivalence_classes=True
        )
        incidence_pairs = [tuple(d) for d in df.values]
        incidence_df = self._E.properties.loc[incidence_pairs]
        ekeys = list(eclasses.keys())
        ekeys = list(set(ekeys).intersection(self.nodes.properties.index))
        if share_properties == True:
            H = _construct_hyp_from_stores(incidence_df)
        else:
            node_ps = PropertyStore(self.nodes.properties.copy(deep=True).loc[ekeys])
            edge_ps = PropertyStore(self.edges.properties.copy(deep=True))
            H = self._construct_hyp_from_stores(
                incidence_df, edge_ps=edge_ps, node_ps=node_ps, name=name
            )
        if return_equivalence_classes == True:
            return H, eclasses
        else:
            return H

    def collapse_nodes_and_edges(
        self,
        name=None,
        use_edge_ids=None,
        use_node_ids=None,
        return_equivalence_classes=False,
    ):
        """
        Returns a new hypergraph by collapsing nodes and edges.

        Parameters
        ----------

        name: str, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes keyed by frozen
            sets of nodes

        use_reps: boolean, optional, default = None
            [DEPRECATED; WILL BE REMOVED IN NEXT RELEASE] Choose a single element from the collapsed elements as a
            representative. If use_reps is True, the new elements are keyed by a tuple of the
            rep and the count.

        return_counts: boolean, optional, default = None
            [DEPRECATED; WILL BE REMOVED IN NEXT RELEASE]

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Collapses the Nodes and Edges of EntitySets. Two nodes(edges) are
        duplicates if their respective memberships(elements) are the same.
        Using this as an equivalence relation, the uids of the nodes(edges)
        are partitioned into equivalence classes. A single member of the
        equivalence class is chosen to represent the class followed by the
        number of members of the class.

        Example
        -------

            >>> data = {'E1': ('a', 'b'), 'E2': ('a', 'b')}
            >>> h = Hypergraph(data)
            >>> h.incidence_dict
            {'E1': ['a', 'b'], 'E2': ['a', 'b']}
            >>> h.collapse_nodes_and_edges().incidence_dict
            {'E1: 2': ['a: 2']}

        """

        if return_equivalence_classes:
            temp, neq = self.collapse_nodes(
                use_uids=use_node_ids, return_equivalence_classes=True
            )
            ntemp, eeq = temp.collapse_edges(
                name=name, use_uids=use_edge_ids, return_equivalence_classes=True
            )
            return ntemp, neq, eeq
        else:
            temp = self.collapse_nodes(name="temp", use_uids=use_node_ids)
            return temp.collapse_edges(name=name, use_uids=use_edge_ids)

    #### restrict_to methods should be handled in the incidence
    #### store and should preserve stores if inplace=True otherwise
    #### deepcopy should be returned and properties will be disconnected
    def restrict_to_nodes(self, nodes, name=None, share_properties=False):
        """New hypergraph gotten by restricting to nodes

        Parameters
        ----------
        nodes : Iterable
            nodeids to restrict to

        Returns
        -------
        : hnx. Hypergraph

        """
        keys = list(set(self._state_dict["labels"]["nodes"]).difference(nodes))
        return self.remove_nodes(keys, name=name, share_properties=share_properties)

    def restrict_to_edges(self, edges, name=None, share_properties=False):
        """New hypergraph gotten by restricting to edges

        Parameters
        ----------
        edges : Iterable
            edgeids to restrict to

        Returns
        -------
        hnx.Hypergraph

        """
        keys = list(set(self._state_dict["labels"]["edges"]).difference(edges))
        return self.remove_edges(keys, name=name, share_properties=share_properties)

    #### This should follow behavior of restrictions
    def remove_edges(self, keys, name=None, share_properties=False):
        if isinstance(keys, list):
            return self.remove(
                keys, level=0, name=name, share_properties=share_properties
            )
        else:
            return self.remove(
                [keys], level=0, name=name, share_properties=share_properties
            )

    def remove_nodes(self, keys, name=None, share_properties=False):
        if isinstance(keys, list):
            return self.remove(
                keys, level=1, name=name, share_properties=share_properties
            )
        else:
            return self.remove(
                [keys], level=1, name=name, share_properties=share_properties
            )

    def remove_incidences(self, keys, name=None, share_properties=False):
        if isinstance(keys, list):
            return self.remove(keys, name=name, share_properties=share_properties)
        else:
            return self.remove([keys], name=name, share_properties=share_properties)

    def remove(self, uid_list, level=None, name=None, share_properties=False):
        """Creates a new hypergraph with nodes and/or edges indexed by keys
        removed. More efficient for creating a restricted hypergraph if the
        restricted set is greater than what is being removed.

        Parameters
        ----------
        uid_list : list
            list of uids from edges, nodes, or incidence pairs(listed as tuples)
        level : None, optional
            Enter 0 to remove edges.
            Enter 1 to remove nodes.
            Enter 2 to remove incidence pairs as tuples
        name : str, optional
            Name of new hypergraph
        share_properties : bool, default = False
            Whether or not to use the same property dataframe as the base hypergraph.
            Sharing means changes to one will effect the other.
            This only applies to the node and edge properties.

        Returns
        -------
        : hnx.Hypergraph

        """

        if level in [0, 1]:
            df = self.properties.copy(deep=True)
            df = df.drop(labels=uid_list, level=level, errors="ignore")
        else:
            df = self.properties.copy(deep=True)
            df = df.drop(labels=uid_list, errors="ignore")
        if share_properties == True:
            return self._construct_hyp_from_stores(df, name=name)
        else:
            edgedf = self._edges.properties.copy(deep=True)
            nodedf = self._nodes.properties.copy(deep=True)
            if level == 0:
                edge_ps = PropertyStore(edgedf.drop(labels=uid_list, errors="ignore"))
                node_ps = PropertyStore(nodedf)
            if level == 1:
                edge_ps = PropertyStore(edgedf)
                node_ps = PropertyStore(nodedf.drop(labels=uid_list, errors="ignore"))
            return self._construct_hyp_from_stores(
                df, edge_ps=edge_ps, node_ps=node_ps, name=name
            )

    #### this should follow behavior of restrictions
    def toplexes(self, name=None):
        """
        Returns a :term:`simple hypergraph` corresponding to self.

        Warning
        -------
        Collapsing is no longer supported inside the toplexes method. Instead
        generate a new collapsed hypergraph and compute the toplexes of the
        new hypergraph.

        Parameters
        ----------
        name: str, optional, default = None
        """

        thdict = {}
        for e in self.edges:
            thdict[e] = self.edges[e]

        tops = []
        for e in self.edges:
            flag = True
            old_tops = list(tops)
            for top in old_tops:
                if set(thdict[e]).issubset(thdict[top]):
                    flag = False
                    break

                if set(thdict[top]).issubset(thdict[e]):
                    tops.remove(top)
            if flag:
                tops += [e]
        return self.restrict_to_edges(tops, name=name)

    #### hypergraph method using linegraph gotten from incidence store
    def is_connected(self, s=1, edges=False):
        """
        Determines if hypergraph is :term:`s-connected <s-connected,
        s-node-connected>`.

        Parameters
        ----------
        s: int, optional, default 1

        edges: boolean, optional, default = False
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

        g = self.get_linegraph(s=s, edges=edges)
        is_connected = None

        try:
            is_connected = nx.is_connected(g)
        except nx.NetworkXPointlessConcept:
            warnings.warn("Graph is null; ")
            is_connected = False

        return is_connected

    def singletons(self):
        """
        Returns a list of singleton edges. A singleton edge is an edge of
        size 1 with a node of degree 1.

        Returns
        -------
        singles : list
            A list of edge uids.
        """

        M, _, cdict = self.incidence_matrix(index=True)
        # which axis has fewest members? if 1 then columns
        idx = np.argmax(M.shape).tolist()
        # we add down the row index if there are fewer columns
        cols = M.sum(idx)
        singles = []
        # index along opposite axis with one entry each
        for c in np.nonzero((cols - 1 == 0))[(idx + 1) % 2]:
            # if the singleton entry in that column is also
            # singleton in its row find the entry
            if idx == 0:
                r = np.argmax(M.getcol(c))
                # and get its sum
                s = np.sum(M.getrow(r))
                # if this is also 1 then the entry in r,c represents a
                # singleton so we want to change that entry to 0 and
                # remove the row. this means we want to remove the
                # edge corresponding to c
                if s == 1:
                    singles.append(cdict[c])
            else:  # switch the role of r and c
                r = np.argmax(M.getrow(c))
                s = np.sum(M.getcol(r))
                if s == 1:
                    singles.append(cdict[r])
        return singles

    def remove_singletons(self, name=None):
        """
        Constructs clone of hypergraph with singleton edges removed.

        Returns
        -------
        new hypergraph : Hypergraph

        """
        singletons = self.singletons()
        if len(singletons) > len(self.edges):
            E = [e for e in self.edges if e not in singletons]
            return self.restrict_to_edges(E, name=name)
        else:
            return self.remove(singletons, level=0, name=name)

    def s_connected_components(self, s=1, edges=True, return_singletons=False):
        """
        Returns a generator for the :term:`s-edge-connected components
        <s-edge-connected component>`
        or the :term:`s-node-connected components <s-connected component,
        s-node-connected component>` of the hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        edges : boolean, optional, default = True
            If True will return edge components, if False will return node
            components
        return_singletons : bool, optional, default = False

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
        sequence of nodes starting with v1 and ending with v2 such that
        pairwise adjacent nodes in the sequence share s edges. If s=1 these
        are the path components of the hypergraph.

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
            Iterator returns sets of uids of the edges (or nodes) in the
            s-edge(node) components of hypergraph.

        """
        g = self.get_linegraph(s, edges=edges)
        for c in nx.connected_components(g):
            if not return_singletons and len(c) == 1:
                continue
            yield c

    def s_component_subgraphs(
        self, s=1, edges=True, return_singletons=False, name=None
    ):
        """

        Returns a generator for the induced subgraphs of s_connected
        components. Removes singletons unless return_singletons is set to True.
        Computed using s-linegraph generated either by the hypergraph
        (edges=True) or its dual (edges = False)

        Parameters
        ----------
        s : int, optional, default 1

        edges : boolean, optional, edges=False
            Determines if edge or node components are desired. Returns
            subgraphs equal to the hypergraph restricted to each set of
            nodes(edges) in the s-connected components or s-edge-connected
            components
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
                yield self.restrict_to_edges(c, name=f"{name or self.name}:{idx}")
            else:
                yield self.restrict_to_nodes(c, name=f"{name or self.name}:{idx}")

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

    def connected_components(self, edges=False):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(edges=edges, return_singletons=True)

    def connected_component_subgraphs(self, return_singletons=True, name=None):
        """
        Same as :meth:`s_component_subgraphs` with s=1. Returns iterator

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(
            return_singletons=return_singletons, name=name
        )

    def components(self, edges=False):
        """
        Same as :meth:`s_connected_components` with s=1, but nodes are returned
        by default. Return iterator.

        See Also
        --------
        s_connected_components
        """
        return self.s_connected_components(s=1, edges=edges)

    def component_subgraphs(self, return_singletons=False, name=None):
        """
        Same as :meth:`s_components_subgraphs` with s=1. Returns iterator.

        See Also
        --------
        s_component_subgraphs
        """
        return self.s_component_subgraphs(
            return_singletons=return_singletons, name=name
        )

    def node_diameters(self, s=1):
        """
        Returns the node diameters of the connected components in hypergraph.

        Parameters
        ----------
        list of the diameters of the s-components and
        list of the s-component nodes
        """
        A, coldict = self.adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_array(A)
        diams = []
        comps = []
        for c in nx.connected_components(G):
            diamc = nx.diameter(G.subgraph(c))
            temp = set()
            for e in c:
                temp.add(coldict[e])
            comps.append(temp)
            diams.append(diamc)
        loc = np.argmax(diams).tolist()
        return diams[loc], diams, comps

    def edge_diameters(self, s=1):
        """
        Returns the edge diameters of the s_edge_connected component subgraphs
        in hypergraph.

        Parameters
        ----------
        s : int, optional, default 1

        Returns
        -------
        maximum diameter : int

        list of diameters : list
            List of edge_diameters for s-edge component subgraphs in hypergraph

        list of component : list
            List of the edge uids in the s-edge component subgraphs.

        """
        A, coldict = self.edge_adjacency_matrix(s=s, index=True)
        G = nx.from_scipy_sparse_array(A)
        diams = []
        comps = []
        for c in nx.connected_components(G):
            diamc = nx.diameter(G.subgraph(c))
            temp = set()
            for e in c:
                temp.add(coldict[e])
            comps.append(temp)
            diams.append(diamc)
        loc = np.argmax(diams).tolist()
        return diams[loc], diams, comps

    def diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between nodes in
        hypergraph

        Parameters
        ----------
        s : int, optional, default 1

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
        Two nodes v_start and v_end are s-walk connected if there is a
        sequence of nodes v_start, v_1, v_2, ... v_n-1, v_end such that
        consecutive nodes are s-adjacent. If the graph is not connected,
        an error will be raised.

        """
        A = self.adjacency_matrix(s=s)
        G = nx.from_scipy_sparse_array(A)
        if nx.is_connected(G):
            return nx.diameter(G)

        raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")

    def edge_diameter(self, s=1):
        """
        Returns the length of the longest shortest s-walk between edges in
        hypergraph

        Parameters
        ----------
        s : int, optional, default 1

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
        Two nodes e_start and e_end are s-walk connected if there is a
        sequence of edges e_start, e_1, e_2, ... e_n-1, e_end such that
        consecutive edges are s-adjacent. If the graph is not connected, an
        error will be raised.

        """
        A = self.edge_adjacency_matrix(s=s)
        G = nx.from_scipy_sparse_array(A)
        if nx.is_connected(G):
            return nx.diameter(G)

        raise HyperNetXError(f"Hypergraph is not s-connected. s={s}")

    def distance(self, source, target, s=1):
        """
        Returns the shortest s-walk distance between two nodes in the
        hypergraph.

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
        g = self.get_linegraph(s=s, edges=False)
        try:
            dist = nx.shortest_path_length(g, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            warnings.warn(f"No {s}-path between {source} and {target}")
            dist = np.inf

        return dist

    def edge_distance(self, source, target, s=1):
        """XX TODO: still need to return path and translate into user defined
        nodes and edges Returns the shortest s-walk distance between two edges
        in the hypergraph.

        Parameters
        ----------
        source : edge.uid or edge
            an edge in the hypergraph

        target : edge.uid or edge
            an edge in the hypergraph

        s : positive integer
            the number of intersections between pairwise consecutive edges

        TODO: add edge weights
        weight : None or string, optional, default = None
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
            An s-walk between edges is a sequence of edges such that
            consecutive pairwise edges intersect in at least s nodes. The
            length of the shortest s-walk is 1 less than the number of edges
            in the path sequence.

            Uses the networkx shortest_path_length method on the graph
            generated by the s-edge_adjacency matrix.

        """
        g = self.get_linegraph(s=s, edges=True)
        try:
            edge_dist = nx.shortest_path_length(g, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            warnings.warn(f"No {s}-path between {source} and {target}")
            edge_dist = np.inf

        return edge_dist

    #### Needs to create stores then hypergraph.
    @classmethod
    #### Need to preserve graph properties in data
    def from_bipartite(cls, B, node_id=1, name=None, **kwargs):
        """
        Static method creates a Hypergraph from a bipartite graph.

        Parameters
        ----------

        B: nx.Graph()
            A networkx bipartite graph. Each node in the graph has a property
            'bipartite' taking the value of 0 or 1 indicating a 2-coloring of
            the graph.

        node_col : int
            bipartite value assigned to graph nodes that will be hypergraph
            edges

        name: hashable, optional

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
            >>> B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), /
                (3, 'c'), (4, 'a')])
            >>> H = Hypergraph.from_bipartite(B, nodes=1)
            >>> H.nodes, H.edges
            # output: (EntitySet(_:Nodes,[1, 2, 3, 4],{}), /
            # EntitySet(_:Edges,['b', 'c', 'a'],{}))

        """

        edges = []
        nodes = []
        for n, d in B.nodes(data=True):
            if d["bipartite"] == node_id:
                nodes.append(n)
            else:
                edges.append(n)

        if not bipartite.is_bipartite_node_set(B, nodes):
            raise HyperNetXError(
                "Error: Method requires a 2-coloring of a bipartite graph."
            )

        elist = []
        for e in list(B.edges):
            if e[0] in edges:
                elist.append([e[0], e[1]])
            else:
                elist.append([e[1], e[0]])
        df = pd.DataFrame(elist)
        return Hypergraph(df, name=name, **kwargs)

    @classmethod
    def from_incidence_matrix(
        cls,
        M,
        name=None,
        **kwargs,
    ):
        """
        Accepts numpy.matrix or scipy.sparse matrix
        """
        mat = coo_matrix(M)
        edges = mat.col
        nodes = mat.row
        weights = mat.data
        df = pd.DataFrame({'edges':edges,'nodes':nodes,'weights':weights})
        return Hypergraph(df,cell_weight_col='weights',name=name,**kwargs)

    @classmethod
    def from_numpy_array(
        cls,
        M,
        node_names=None,
        edge_names=None,
        name=None,
        key=None,
        **kwargs,
    ):
        """
        Create a hypergraph from a real valued matrix represented as a 2 dimensional numpy array.
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
        # Create names for nodes and edges
        # Validate the size of the node and edge arrays

        M = np.array(M)
        if len(M.shape) != 2:
            raise HyperNetXError("Input requires a 2 dimensional numpy array")
        # apply boolean key if available
        if key is not None:
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

        df = pd.DataFrame(M, columns=edgenames, index=nodenames)
        return Hypergraph.from_incidence_dataframe(df, name=name)

    @classmethod
    def from_incidence_dataframe(
        cls,
        df,
        columns=None,
        rows=None,
        name=None,
        fillna=0,
        transpose=False,
        transforms=[],
        key=None,
        return_only_dataframe=False,
        **kwargs,
    ):
        """
        Create a hypergraph from a Pandas Dataframe object, which has values equal
        to the incidence matrix of a hypergraph. Its index will identify the nodes
        and its columns will identify its edges.

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
            a real value to place in empty cell, all-zero columns will not
            generate an edge.

        transpose : (optional) bool, default = False
            option to transpose the dataframe, in this case df.Index will
            identify the edges and df.columns will identify the nodes, transpose is
            applied before transforms and key

        transforms : (optional) list, default = []
            optional list of transformations to apply to each column,
            of the dataframe using pd.DataFrame.apply().
            Transformations are applied in the order they are
            given (ex. abs). To apply transforms to rows or for additional
            functionality, consider transforming df using pandas.DataFrame
            methods prior to generating the hypergraph.

        key : (optional) function, default = None
            boolean function to be applied to dataframe. will be applied to
            entire dataframe.

        return_only_dataframe : (optional) bool, default = False
            to use the incidence_dataframe with cell_properties or properties, set this
            to true and use it as the setsystem in the Hypergraph constructor.

        See also
        --------
        from_numpy_array


        Returns
        -------
        : Hypergraph

        """

        if not isinstance(df, pd.DataFrame):
            raise HyperNetXError("Error: Input object must be a pandas dataframe.")

        if columns:
            df = df[columns]
        if rows:
            df = df.loc[rows]

        df = df.fillna(fillna)
        if transpose:
            df = df.transpose()

        for t in transforms:
            df = df.apply(t)
        if key:
            mat = key(df.values) * 1
        else:
            mat = df.values * 1

        cols = df.columns
        rows = df.index
        CM = coo_matrix(mat)
        c1 = CM.row
        c1 = [rows[c1[idx]] for idx in range(len(c1))]
        c2 = CM.col
        c2 = [cols[c2[idx]] for idx in range(len(c2))]
        c3 = CM.data

        dfnew = pd.DataFrame({"edges": c2, "nodes": c1, "weight": c3})
        if return_only_dataframe == True:
            return dfnew
        else:
            return Hypergraph(dfnew, cell_weight_col="weight", name=name, **kwargs)

    def __add__(self, other):
        """
        Concatenate incidences from two hypergraphs, removing duplicates and
        dropping duplicate property data in the order of addition.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
        Hypergraph

        """
        return self.sum(other)

    def __sub__(self, other):
        """
        Concatenate incidences from two hypergraphs, removing duplicates and
        dropping duplicate property data in the order of addition.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
        Hypergraph

        """
        return self.difference(other)

    def sum(self, other):
        """
        Concatenate incidences from two hypergraphs, removing duplicates and
        dropping duplicate property data in the order of addition.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
        Hypergraph

        """
        df = self.dataframe
        odf = other.dataframe
        ndf = pd.concat([df, odf]).groupby(["edges", "nodes"]).agg("first")
        edf = self.edges.dataframe
        oedf = other.edges.dataframe
        nedf = pd.concat([edf, oedf]).groupby("uid").agg("first")
        nddf = self.nodes.dataframe
        onddf = other.nodes.dataframe
        nnddf = pd.concat([nddf, onddf]).groupby("uid").agg("first")
        return self._construct_hyp_from_stores(
            ndf, edge_ps=PropertyStore(nedf), node_ps=PropertyStore(nnddf)
        )

    def difference(self, other):
        """
        Concatenate incidences from two hypergraphs, removing duplicates and
        dropping duplicate property data in the order of addition.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
        Hypergraph

        """
        df = self.incidences.properties
        odf = other.incidences.properties
        ndf = df.loc[~df.index.isin(odf.index.tolist())]
        edf = self.edges.properties
        oedf = other.edges.properties
        nedf = edf.loc[~edf.index.isin(oedf.index.tolist())]
        nddf = self.nodes.properties
        onddf = other.nodes.properties
        nnddf = nddf.loc[~nddf.index.isin(onddf.index.tolist())]
        return self._construct_hyp_from_stores(
            ndf, edge_ps=PropertyStore(nedf), node_ps=PropertyStore(nnddf)
        )
