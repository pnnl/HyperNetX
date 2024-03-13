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

from hypernetx.classes import EntitySet
from hypernetx.exception import HyperNetXError
from hypernetx.utils.decorators import warn_nwhy
from hypernetx.classes.helpers import merge_nested_dicts, dict_depth

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
        default_cell_weight: float = 1.0,    ### we will no longer support a sequence
        edge_col: str | int = 0,
        node_col: str | int = 1,
        cell_weight_col: Optional[str | int] = "weight",

        cell_properties: Optional[
            Sequence[str | int] | Mapping[T, Mapping[T, Mapping[str, Any]]]
        ] = None,
        misc_cell_properties_col: Optional[str | int] = None,
        aggregateby: str | dict[str, str] = "first",

        ### Format for properties can be either a dataframe indexed on uid
        ### or with first column equal to uid or a dictionary
        ### use these for a single properties list
        properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        prop_uid_col : str | int | None = None, ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_properties_col: Optional[str | int] = None,
        weight_prop_col: str | int = "weight",
        default_weight: float = 1.0,   

        ### these are just for properties on the edges - ignored if properties exists
        edge_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        edge_uid_col : str | int | None, ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_edge_properties_col: Optional[str | int] = None,
        edge_weight_prop_col: str | int = "weight",
        default_edge_weight: float = 1.0,

        ### these are just for properties on the nodes - ignored if properties exists
        node_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        node_uid_col : str | int | None, ### this means the index will be used for uid
        ### How do we know which column to use for uid
        misc_node_properties_col: Optional[str | int] = None,
        node_weight_prop_col: str | int = "weight",
        default_node_weight: float = 1.0,

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
            'DataFrame' : dataframe_factory_method,
            'dict' : dict_factory_method,
            'list' : list_factory_method,
        }
    
    ## dataframe_factory_method(setsystem_df,index_cols=[edge_col,node_col],weight_col,default_weight,misc_properties,aggregate_by)
    ## dataframe_factory_method(edge_properties_df,index_cols=[edge_uid_col],edge_weight_col,default_edge_weight,misc_edge_properties)

        setsystem_type = type(setsystem)
        if setsystem_type in type_dict:
            df = type_dict[setsystem_type](setsystem,
                                           index_cols=[edge_col,node_col],
                                           cell_weight_col = cell_weight_col,
                                           default_cell_weight=default_cell_weight,
                                           misc_cell_properties_col=misc_cell_properties_col,
                                           aggregate_by=aggregate_by)
    ## dataframe_factory_method(edf,index_cols=[uid_col],weight_col,default_weight,misc_properties)
            ## multi index set by uid_cols = [edge_col,node_col]
            incidence_store = IncidenceStore(df.index)
            incidence_propertystore = PropertyStore(df,index=True) 
            self._E = HypergraphView(incidence_store,2,incidence_propertystore)
            ## if no properties PropertyStore should store in the most efficient way
        else:
            raise HyperNetXError("setsystem data type not supported")
        ## check if there is another constructor they could use.

        if properties is not None:
            property_type = type(properties)
            if property_type in type_dict:
                dfp = type_dict[property_type](properties,
                                               index_cols=[prop_uid_col],
                                               property_weight_col=property_weight_col,
                                               default_property_weight=default_property_weight,
                                               misc_properties_col=misc_properties_col)
                all_propertystore = PropertyStore(dfp)
                self.edges = HypergraphView(incidence_store,0,all_propertystore)
                self.nodes = HypergraphView(incidence_store,1,all_propertystore)
        else:
            if edge_properties is not None:
                edge_property_type = type(edge_properties)
                if edge_property_type in type_dict:
                    edfp = type_dict[edge_property_type](edge_properties,index_cols=[edge_uid_col],edge_weight_col,default_edge_weight,misc_edge_properties)
                    edge_propertystore = PropertyStore(edfp)
                else:
                    edge_propertystore = PropertyStore()       
                self.edges = HypergraphView(incidence_store,0,edge_propertystore) 
            if node_properties is not None:
                node_property_type = type(node_properties)
                if node_property_type in type_dict:
                    ndfp = type_dict[node_property_type](node_properties,index_cols=[node_uid_col],node_weight_col,default_node_weight,misc_node_properties)
                    node_propertystore = PropertyStore(ndfp)
                else:
                    node_propertystore = PropertyStore()       
                self.nodes = HypergraphView(incidence_store,0,node_propertystore)     
                    
                    

        self._dataframe = self.dataframe()
        self._set_default_state()
        self._dict_.update(locals())

    @property
    def edges(self):
        """
        Object associated with self._edges.

        Returns
        -------
        EntitySet
        """
        return self._edges

    @property
    def nodes(self):
        """
        Object associated with self._nodes.

        Returns
        -------
        EntitySet
        """
        return self._nodes

    @property
    def dataframe(self):
        """Returns dataframe of incidence pairs and their properties.

        Returns
        -------
        pd.DataFrame
        """
        return self._E.dataframe

    @property ### TBD
    def properties(self):
        """Returns dataframe of edge and node properties.

        Returns
        -------
        pd.DataFrame or Dictionary?
        """
        ## Concatenate self._edges and self._nodes into a dataframe


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
        return self._E.to_dict()  ### should this call the incidence store directly?

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

    def __getitem__(self, node):
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
        return self._E.properties((edge,node),prop_name=prop_name)  ### get_property from hyp_view

    def get_properties(self, id, level=None, prop_name=None):
        """Returns an object's specific property or all properties
        ### Change to level is 0,1,2 and call props from correct store
        Parameters
        ----------
        id : hashable
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

        if level == None or level == 0:
            store = self._edges
        else:
            store = self._nodes
        if prop_name is None: ## rewrite for edges and nodes.
            return self.store.get_properties(id)
        else:
            return self.store.get_property(id, prop_name)

    @warn_nwhy
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

        if edges: ### Amaplist needs a dictionary returned for properties.
            A, Amap = self.edge_adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self._edges.properties(k)) for k in Amap]
        else:
            A, Amap = self.adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self._nodes.properties(k)) for k in Amap]

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

    def _set_default_state(self,empty=False):
        """Populate state_dict with default values
        This may change with HypegraphView since data is no
        longer required to be in a specific structure
        """
        self._state_dict = {}

        self._state_dict["dataframe"] = df = self.dataframe()

        if empty:
            self._state_dict["labels"] = {
                "edges": np.array([]),
                "nodes": np.array([])
                }
            self._state_dict["data"] = np.array([[],[]])
#### Do we need to set types to category for use of encoders in pandas?
        else:
            self._state_dict["labels"] = {
                "edges": np.array(df[self._edge_col].cat.categories),
                "nodes": np.array(df[self._node_col].cat.categories),
            }
            self._state_dict["data"] = np.array(
                [df[self._edge_col].cat.codes, df[self._node_col].cat.codes], dtype=int
            ).T


        self._state_dict["snodelg"] = dict()  ### s: nx.graph
        self._state_dict["sedgelg"] = dict()
        self._state_dict["neighbors"] = defaultdict(dict)  ### s: {node: neighbors}
        self._state_dict["edge_neighbors"] = defaultdict(dict)  ### s: {edge: edge_neighbors}
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

    def number_of_nodes(self, nodeset=None):
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

    def number_of_edges(self, edgeset=None):
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

    def incidence_matrix(self, weights=False, index=False):
        """
        An incidence matrix for the hypergraph indexed by nodes x edges.

        Parameters
        ----------
        weights : bool, default =False
            If False all nonzero entries are 1.
            If True and self.static all nonzero entries are filled by
            self.edges.cell_weights dictionary values.

        index : boolean, optional, default = False
            If True return will include a dictionary of node uid : row number
            and edge uid : column number

        Returns
        -------
        incidence_matrix : scipy.sparse.csr.csr_matrix or np.ndarray

        row_index : list
            index of node ids for rows

        col_index : list
            index of edge ids for columns

        """
        sdkey = "incidence_matrix"
        if weights:
            sdkey = "weighted_" + sdkey

        if sdkey in self._state_dict:
            M = self._state_dict[sdkey]
        else:
            df = self.dataframe
            data_cols = [self._node_col, self._edge_col]
            if weights == True:
                data = df[self._cell_weight_col].values
                M = csr_matrix(
                    (data, tuple(np.array(df[col].cat.codes) for col in data_cols))
                )
            else:
                M = csr_matrix(
                    (
                        [1] * len(df),
                        tuple(np.array(df[col].cat.codes) for col in data_cols),
                    )
                )
            self._state_dict[sdkey] = M

        if index == True:
            rdx = self.dataframe[self._node_col].cat.categories
            cdx = self.dataframe[self._edge_col].cat.categories

            return M, rdx, cdx
        else:
            return M

    def incidence_dataframe(self, prop_name = 'weight'):
        """
        pivot table from dataframe for self._E, specifying
        cell value by property from cell properties

        Parameters
        ----------
        prop_name : str, optional
            _description_, by default 'weight'
        """
        pass

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

    def bipartite(self):
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
        B = nx.Graph()
        nodes = self._state_dict["labels"]["nodes"]
        edges = self._state_dict["labels"]["edges"]
        B.add_nodes_from(self.edges, bipartite=0)
        B.add_nodes_from(self.nodes, bipartite=1)
        B.add_edges_from([(v, e) for e in self.edges for v in self.edges[e]])
        return B

    def dual(self, name=None, switch_names=True):
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
        dfp = deepcopy(self.edges.properties)
        dfp = dfp.reset_index()
        dfp.level = dfp.level.apply(lambda x: 1 * (x == 0))
        dfp = dfp.set_index(["level", "id"])

        edge, node, wt = self._edge_col, self._node_col, self._cell_weight_col
        df = deepcopy(self.dataframe)
        cprops = [col for col in df.columns if not col in [edge, node, wt]]

        df[[edge, node]] = df[[node, edge]]
        if switch_names == True and not (
            self._edge_col == "edges" and self._node_col == "nodes"
        ):
            # if switch_names == False or (self._edge_col == 'edges' and self._node_col == 'nodes'):
            df = df.rename(columns={edge: self._node_col, node: self._edge_col})
            node = self._edge_col
            edge = self._node_col

        return Hypergraph(
            df,
            edge_col=edge,
            node_col=node,
            cell_weight_col=wt,
            cell_properties=cprops,
            properties=dfp,
            name=name,
        )

###### Collapse methods now handled completely in the incidence store
    def collapse_edges(
        self,
        name=None,
        return_equivalence_classes=False,
        use_reps=None,
        return_counts=None,
    ):
        """
        Constructs a new hypergraph gotten by identifying edges containing the
        same nodes

        Parameters
        ----------
        name : hashable, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes keyed by frozen
            sets of nodes

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
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps ane return_counts are no longer supported keyword
            arguments and will throw an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a
            string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        temp = self.edges.collapse_identical_elements(
            return_equivalence_classes=return_equivalence_classes
        )

        if return_equivalence_classes:
            return Hypergraph(temp[0].incidence_dict, name), temp[1]

        return Hypergraph(temp.incidence_dict, name)

    def collapse_nodes(
        self,
        name=None,
        return_equivalence_classes=False,
        use_reps=None,
        return_counts=None,
    ) -> Hypergraph:
        """
        Constructs a new hypergraph gotten by identifying nodes contained by
        the same edges

        Parameters
        ----------
        name: str, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of node equivalence classes keyed by frozen
            sets of edges

        use_reps : boolean, optional, default = None
            [DEPRECATED; WILL BE REMOVED IN NEXT RELEASE] Choose a single element from the
            collapsed nodes as uid for the new node, otherwise uses a frozen
            set of the uids of nodes in the equivalence class. If use_reps is True the new nodes have uids given by a
            tuple of the rep and the count

        return_counts: boolean, optional, default = None
            [DEPRECATED; WILL BE REMOVED IN NEXT RELEASE]

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Two nodes are identified if their respective memberships are the same.
        Using this as an equivalence relation, the uids of the nodes are
        partitioned into equivalence classes. A single member of the
        equivalence class is chosen to represent the class followed by the
        number of members of the class.

        Example
        -------

            >>> data = {'E1': ('a', 'b'), 'E2': ('a', 'b')}))
            >>> h = Hypergraph(data)
            >>> h.collapse_nodes().incidence_dict
            {'E1': ['a: 2'], 'E2': ['a: 2']}
        """
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps and return_counts are no longer supported keyword arguments and will throw
            an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        temp = self.dual().edges.collapse_identical_elements(
            return_equivalence_classes=return_equivalence_classes
        )

        if return_equivalence_classes:
            return Hypergraph(temp[0].incidence_dict).dual(), temp[1]

        return Hypergraph(temp.incidence_dict, name).dual()

    def collapse_nodes_and_edges(
        self,
        name=None,
        return_equivalence_classes=False,
        use_reps=None,
        return_counts=None,
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
        if use_reps is not None or return_counts is not None:
            msg = """
            use_reps and return_counts are no longer supported keyword
            arguments and will throw an error in the next release.
            collapsed hypergraph automatically names collapsed objects by a
            string "rep:count"
            """
            warnings.warn(msg, DeprecationWarning)

        if return_equivalence_classes:
            temp, neq = self.collapse_nodes(
                name="temp", return_equivalence_classes=True
            )
            ntemp, eeq = temp.collapse_edges(name=name, return_equivalence_classes=True)
            return ntemp, neq, eeq

        temp = self.collapse_nodes(name="temp")
        return temp.collapse_edges(name=name)

#### restrict_to methods should be handled in the incidence
#### store and should preserve stores if inplace=True otherwise
#### deepcopy should be returned and properties will be disconnected
    def restrict_to_nodes(self, nodes, name=None):
        """New hypergraph gotten by restricting to nodes

        Parameters
        ----------
        nodes : Iterable
            nodeids to restrict to

        Returns
        -------
        : hnx. Hypergraph

        """
        keys = set(self._state_dict["labels"]["nodes"]).difference(nodes)
        return self.remove(keys, level=1)

    def restrict_to_edges(self, edges, name=None):
        """New hypergraph gotten by restricting to edges

        Parameters
        ----------
        edges : Iterable
            edgeids to restrict to

        Returns
        -------
        hnx.Hypergraph

        """
        keys = set(self._state_dict["labels"]["edges"]).difference(edges)
        return self.remove(keys, level=0)

#### This should follow behavior of restrictions
    def remove_edges(self, keys, name=None):
        return self.remove(keys, level=0, name=name)

    def remove_nodes(self, keys, name=None):
        return self.remove(keys, level=1, name=name)

    def remove(self, keys, level=None, name=None):
        """Creates a new hypergraph with nodes and/or edges indexed by keys
        removed. More efficient for creating a restricted hypergraph if the
        restricted set is greater than what is being removed.

        Parameters
        ----------
        keys : list | tuple | set | Hashable
            node and/or edge id(s) to restrict to
        level : None, optional
            Enter 0 to remove edges with ids in keys.
            Enter 1 to remove nodes with ids in keys.
            If None then all objects in nodes and edges with the id will
            be removed.
        name : str, optional
            Name of new hypergraph

        Returns
        -------
        : hnx.Hypergraph

        """
        rdfprop = self.properties.copy()
        rdf = self.dataframe.copy()
        if isinstance(keys, (list, tuple, set)):
            nkeys = keys
        elif isinstance(keys, Hashable):
            nkeys = list()
            nkeys.append(keys)
        else:
            raise TypeError("`keys` parameter must be list | tuple | set | Hashable")
        if level == 0:
            kdx = set(nkeys).intersection(set(self._state_dict["labels"]["edges"]))
            for k in kdx:
                rdfprop = rdfprop.drop((0, k))
            rdf = rdf.loc[~(rdf[self._edge_col].isin(kdx))]
        elif level == 1:
            kdx = set(nkeys).intersection(set(self._state_dict["labels"]["nodes"]))
            for k in kdx:
                rdfprop = rdfprop.drop((1, k))
            rdf = rdf.loc[~(rdf[self._node_col].isin(kdx))]
        else:
            rdfprop = rdfprop.reset_index()
            kdx = set(nkeys).intersection(rdfprop.id.unique())
            rdfprop = rdfprop.set_index("id")
            rdfprop = rdfprop.drop(index=kdx)
            rdf = rdf.loc[~(rdf[self._edge_col].isin(kdx))]
            rdf = rdf.loc[~(rdf[self._node_col].isin(kdx))]

        return Hypergraph(
            setsystem=rdf,
            edge_col=self._edge_col,
            node_col=self._node_col,
            cell_weight_col=self._cell_weight_col,
            misc_cell_properties_col=self.edges._misc_cell_props_col,
            properties=rdfprop,
            misc_properties_col=self.edges._misc_props_col,
            name=name,
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
        G = nx.from_scipy_sparse_matrix(A)
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
        G = nx.from_scipy_sparse_matrix(A)
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

#### pivot table on dataframe gotten from incidence store
    def incidence_dataframe(
        self, sort_rows=False, sort_columns=False, cell_weights=True
    ):
        """
        Returns a pandas dataframe for hypergraph indexed by the nodes and
        with column headers given by the edge names.

        Parameters
        ----------
        sort_rows : bool, optional, default =True
            sort rows based on hashable node names
        sort_columns : bool, optional, default =True
            sort columns based on hashable edge names
        cell_weights : bool, optional, default =True

        """

        ## An entity dataframe is already an incidence dataframe.
        df = self._E.dataframe.pivot(
            index=self._E._data_cols[1],
            columns=self._E._data_cols[0],
            values=self._E._cell_weight_col,
        ).fillna(0)

        if sort_rows:
            df = df.sort_index("index")
        if sort_columns:
            df = df.sort_index("columns")
        if not cell_weights:
            df[df > 0] = 1
            df = df.astype(int)

        return df

#### Needs to create stores then hypergraph.
    @classmethod
    @warn_nwhy #### Need to preserve graph properties
    def from_bipartite(cls, B, set_names=("edges", "nodes"), name=None, **kwargs):
        """
        Static method creates a Hypergraph from a bipartite graph.

        Parameters
        ----------

        B: nx.Graph()
            A networkx bipartite graph. Each node in the graph has a property
            'bipartite' taking the value of 0 or 1 indicating a 2-coloring of
            the graph.

        set_names: iterable of length 2, optional, default = ['edges','nodes']
            Category names assigned to the graph nodes associated to each
            bipartite set

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
            >>> H = Hypergraph.from_bipartite(B)
            >>> H.nodes, H.edges
            # output: (EntitySet(_:Nodes,[1, 2, 3, 4],{}), /
            # EntitySet(_:Edges,['b', 'c', 'a'],{}))

        """

        edges = []
        nodes = []
        for n, d in B.nodes(data=True):
            if d["bipartite"] == 1:
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
        df = pd.DataFrame(elist, columns=set_names)
        return Hypergraph(df, name=name, **kwargs)


    @classmethod
    def from_incidence_matrix(
        cls,
        M,
        node_names=None,
        edge_names=None,
        node_label="nodes",
        edge_label="edges",
        name=None,
        key=None,
        **kwargs,
    ):
        """
        Same as from_numpy_array.
        """
        return Hypergraph.from_numpy_array(
            M,
            node_names=node_names,
            edge_names=edge_names,
            node_label=node_label,
            edge_label=edge_label,
            name=name,
            key=key,
        )

    @classmethod
    @warn_nwhy
    def from_numpy_array(
        cls,
        M,
        node_names=None,
        edge_names=None,
        node_label="nodes",
        edge_label="edges",
        name=None,
        key=None,
        **kwargs,
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
    @warn_nwhy
    def from_incidence_dataframe(
        cls,
        df,
        columns=None,
        rows=None,
        edge_col: str = "edges",
        node_col: str = "nodes",
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

        dfnew = pd.DataFrame({edge_col: c2, node_col: c1, "cell_weights": c3})
        if return_only_dataframe == True:
            return dfnew
        else:
            return Hypergraph(
                dfnew,
                edge_col=edge_col,
                node_col=node_col,
                weights="cell_weights",
                name=None,
            )
