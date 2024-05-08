# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, Any, TypeVar, Union, Mapping

from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix, csr_matrix

from hypernetx.exception import HyperNetXError
from hypernetx.classes.factory import (
    dataframe_factory_method,
    dict_factory_method,
    list_factory_method,
    ndarray_factory_method,
)
from hypernetx.classes.incidence_store import IncidenceStore
from hypernetx.classes.property_store import PropertyStore
from hypernetx.classes.hyp_view import HypergraphView

warnings.filterwarnings("default", category=DeprecationWarning)

__all__ = ["Hypergraph"]

T = TypeVar("T", bound=Union[str, int])


class Hypergraph:
    """
    Parameters
    ----------

    setsystem : (optional) pandas.DataFrame, dict of iterables, dict of dicts,
        list of iterables, numpy.ndarray, default = None
        See SetSystem below for additional setsystem requirements.

    edge_col : (optional) str | int, default = 0
        column index (or name) in pandas.DataFrame,
        used for (hyper)edge ids. Only used when setsystem is a
        pandas.DataFrame

    node_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe,
        used for node ids. Only used when setsystem is a
        pandas.DataFrame

    cell_weight_col : (optional) str | int, default = "weight"
        column index (or name) in pandas.DataFrame used for
        referencing cell weights. For a dict of dicts it will be
        used as a key in the nested dictionary of properties.
        These are the same as edge dependent node weights and
        will populate the incidence matrix when `weights=True`.

    default_cell_weight : (optional) int | float, default = 1
        All incidence pairs in the Hypergraph are assigned a
        default weight if weight is not specified in the setsystem.

    misc_cell_properties_col : (optional) str | int, default = None
        Used for Pandas Dataframe with one column containing dictionaries of
        properties. Useful if objects have diverse property sets.
        Ignored for other setsystem types.

    properties : (optional) pd.DataFrame | dict, default = None
        Concatenation/union of edge_properties and node_properties.
        By default, the object id is used and should be the first column of
        the dataframe, or key in the dict. If there are nodes and edges
        with the same ids but distinct properties then separate them and
        use the edge_properties and node_properties keywords.

    weight_prop_col : (optional) str, default = None
        Name of property in properties to use for weight

    default_weight : (optional) int | float, default = 1
        Used when weight property is missing or undefined

    edge_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with edge ids.
        If a dataframe, the first column must be the names of the edges.
        First column of dataframe or keys of dict link to edge ids in
        setsystem.

    edge_weight_prop_col : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    default_edge_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    node_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with node ids.
        If a dataframe, the first column must be the names of the nodes.
        First column of dataframe or keys of dict link to nodes ids in
        setsystem.

    node_weight_prop_col : (optional) str, default = None,
        Name of property in node_properties to use for weight.

    default_node_weight : (optional) int | float, default = 1
        Used when node weight property is missing or undefined

    misc_properties_col : (optional) str | int, default = None
        Used for properties, edge_properties, and node_properties
        Pandas Dataframes with one column containing dictionaries of
        properties. Useful if objects have diverse property sets.
        Ignored for other setsystem types.

    name : (optional) str, default = None
        Name assigned to hypergraph


    ======================
    Hypergraphs in HNX 2.3
    ======================

    An hnx.Hypergraph H = (V,E) references a pair of disjoint sets:
    V = nodes (vertices) and E = (hyper)edges.

    HNX allows for multi-edges by distinguishing edges by
    their identifiers instead of their contents. For example, if
    V = {1,2,3} and E = {e1,e2,e3},
    where e1 = {1,2}, e2 = {1,2}, and e3 = {1,2,3},
    the edges e1 and e2 contain the same set of nodes and yet
    are distinct and are distinguishable within H = (V,E).

    New as of version 2.3, HNX provides methods to easily store and
    access additional metadata such as cell, edge, and node weights.
    Metadata associated with all edges, nodes, and (edge,node) incidence
    pairs stored in the hypergraph are viewable using: ::

        >>> H.edges.to_dataframe
        >>> H.nodes.to_dataframe
        >>> H.incidences.to_dataframe

    The fundamental object needed to create a hypergraph is a **setsystem**. The
    setsystem defines the many-to-many relationships between edges and nodes in
    the hypergraph. Properties for the incidence pairs are defined within
    the setsystem. Edge and node properties are defined with separate,
    Pandas DataFrames or dictionaries.

    At this time, a hypergraph is defined by its relationships. While the
    nodes and edges are distinct objects with their own properties, it is only
    when they are in relationship that they are viewable within the hypergraph
    structure. This means that hypergraph metrics and combinatorics do not
    use "isolated" nodes or "empty" edges. For example, while the node_properties could
    contain any number of node identifiers, only nodes belonging to an edge
    in the hypergraph are counted when computing the size and shape of the
    hypergraph.

    SetSystems
    ----------
    There are five types of setsystems currently accepted by the library.

    1.  **list of iterables** : Barebones hypergraph uses Pandas default
        indexing to generate hyperedge ids. Elements must be hashable.: ::

        >>> H = Hypergraph([{v1,v2},{v1,v2},{v1,v2,v3}])

    2.  **dictionary of iterables** : the most basic way to express many-to-many
        relationships providing edge ids. The elements of the iterables must be
        hashable): ::

        >>> H = Hypergraph({'e1':[v1,v2],'e2':[v1,v2],'e3':[v1,v2,v3]})

    3.  **dictionary of dictionaries**  : allows cell properties to be assigned
        to a specific (edge, node) incidence. This is particularly useful when
        there are variable length dictionaries assigned to each pair: ::

        >>> d = {'e1':{ v1: {'w':0.5, 'name': 'related_to'},
        >>>             v2: {'w':0.1, 'name': 'related_to',
        >>>                 'startdate': '05.13.2020'}},
        >>>      'e2':{ v1: {'w':0.52, 'name': 'owned_by'},
        >>>             v2: {'w':0.2}},
        >>>      'e3':{ v1: {'w':0.5, 'name': 'related_to'},
        >>>             v2: {'w':0.2, 'name': 'owner_of'},
        >>>             v3: {'w':1, 'type': 'relationship'}}

        >>> H = Hypergraph(d, cell_weight_col='w')

    4.  **pandas.DataFrame** For large datasets and for datasets with edge
        dependent properties it is most efficient to construct a hypergraph
        directly from a pandas.DataFrame.
        The edges column and nodes column may be specified. By default
        they are the first two columns of the dataframe.
        Cell properties shared by all incidence pairs can be placed in their own
        column of the dataframe. Variable length dictionaries of cell properties
        particular to only some of the incidence pairs may be placed in a single
        column of the dataframe. Representing the data above as a dataframe df:

        +-----------+-----------+-----------+-----------------------------------+
        |   edges   |   nodes   |   w       |  col3                             |
        +-----------+-----------+-----------+-----------------------------------+
        |   e1      |   v1      |   0.5     | {'name':'related_to'}             |
        +-----------+-----------+-----------+-----------------------------------+
        |   e1      |   v2      |   0.1     | {"name":"related_to",             |
        |           |           |           |  "startdate":"05.13.2020"}        |
        +-----------+-----------+-----------+-----------------------------------+
        |   e2      |   v1      |   0.52    | {"name":"owned_by"}               |
        +-----------+-----------+-----------+-----------------------------------+
        |   e2      |   v2      |   0.2     |                                   |
        +-----------+-----------+-----------+-----------------------------------+
        |   ...     |   ...     |   ...     | {...}                             |
        +-----------+-----------+-----------+-----------------------------------+

        The first row (header row) of the dataframe is used to reference each
        column. ::

        >>> H = Hypergraph(df,edge_col=0,node_col=1,
        >>>                 cell_weight_col="w",misc_cell_properties="col3")

    5.  **numpy.ndarray** The array must have shape (N,2) for some positive
        integer N. The array describes the incidence tuples for the
        hypergraph. By default the first element of the tuple will reference
        the edge uid and the second the node uid. By specifying edge_col = 1
        and node_col = 0 this can be reversed. The default weight can be assigned
        but other cell attributes will have to be assigned to each individual tuple
        after the hypegraph as been created. If more cell properties are
        available on creation, consider using a Pandas DataFrame.

        >>> arr = np.array([['e1','v1'],['e1','v2'],['e2','v1'],['e2','v3']])
        >>> H = Hypergraph(arr)

    Edge and Node Properties
    ------------------------
    Properties specific to a single edge or node are passed through the
    keywords: **edge_properties, node_properties, or properties**.
    Properties may be passed as dataframes or dictionaries.
    The first column or index of the dataframe or the keys of the dictionary
    correspond to the edge and/or node identifiers.
    If identifiers are shared among edges and nodes, or are distinct
    for edges and nodes, properties may be combined into a single
    object and passed to the **properties** keyword. For example:

        +-----------+-----------+---------------------------------------+
        |   uid     |   weight  |   properties                          |
        +-----------+-----------+---------------------------------------+
        |   e1      |   5.0     |   {'type':'event'}                    |
        +-----------+-----------+---------------------------------------+
        |   e2      |   0.52    |   {"name":"owned_by"}                 |
        +-----------+-----------+---------------------------------------+
        |   ...     |   ...     |   {...}                               |
        +-----------+-----------+---------------------------------------+
        |   1       |   1.2     |   {'color':'red'}                     |
        +-----------+-----------+---------------------------------------+
        |   2       |   .003    |   {'name':'Fido','color':'brown'}     |
        +-----------+-----------+---------------------------------------+
        |   3       |   1.0     |    {}                                 |
        +-----------+-----------+---------------------------------------+

    A properties dictionary should have the format: ::

        dp = {uid1 : {prop1:val1, prop2,val2,...}, uid2 : ... }


    Weights
    -------
    The default key for the incidence pair, edge and node weights is "weight".
    The default values are all set to 1.
    The default key and values may be reset using the keywords:
    `cell_weight_col`, `default_cell_weight`, `weight_col`, `default_weight`,
    `edge_weight_col`, `default_edge_weight`, `node_weight_col`, `default_node_weight`.
    """

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
        misc_cell_properties_col: Optional[str | int] = None,
        aggregate_by: str | dict[str, str] = "first",
        ### Format for properties can be either a dataframe indexed on uid
        ### or with first column equal to uid or a dictionary
        ### use these for a single properties list
        properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        ### How do we know which column to use for uid
        misc_properties_col: Optional[str | int] = None,
        weight_prop_col: str | int = "weight",
        default_weight: float | int = 1,
        ### these are just for properties on the edges - ignored if properties exists
        edge_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        ### How do we know which column to use for uid
        misc_edge_properties_col: Optional[str | int] = None,
        edge_weight_prop_col: str | int = "weight",
        default_edge_weight: float | int = 1,
        ### these are just for properties on the nodes - ignored if properties exists
        node_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        misc_node_properties_col: Optional[str | int] = None,
        node_weight_prop_col: str | int = "weight",
        default_node_weight: float | int = 1,
        name: Optional[str] = None,
        **kwargs,
    ):

        #### Use a Factory Method to create 5 stores
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
            "ndarray": ndarray_factory_method,
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
        : HypergraphView
        """
        return self._edges

    @property
    def nodes(self):
        """
        Object associated with nodes.

        Returns
        -------
        : HypergraphView
        """
        return self._nodes

    @property
    def incidences(self):
        """
        Object associated with incidence pairs

        Returns
        -------
        : HypergraphView
        """
        return self._E

    @property
    def dataframe(self):
        """Returns dataframe of incidence properties
        as dataframe with edges and nodes in columns.

        Returns
        -------
        : pandas.DataFrame
        """
        df = self._E.properties.reset_index()
        return df

    @property
    def properties(self):
        """Returns incidence properties

        Returns
        -------
        : pandas.DataFrame
        """
        return self._E.properties

    def incidence_matrix(self, index=False, weights=False):
        """
        A sparse matrix indicating the existence of an incidence pair
        in the hypergraph. Each row corresponds to a node v and each column
        corresponds to an edge e. The entry corresponding to (row v, col e)
        is nonzero if v is an element of e. If weights = True then the value
        equals the weight given in the hypergraph incidence properties for
        the incidence pair (e,v). Otherwise the value is 1.

        Parameters
        ----------
        index : bool, optional, default = False
            If True will return the nodes and edges ordered to
            correspond to the rows and columns of the matrix
        use_weights : bool, optional, default = False
            If True will use the incidence weights corresponding to
            the row and column of the entry

        Returns
        -------
        : scipy.sparse.csr_matrix

        """
        e, n = self._state_dict["data"].T
        if weights == True:
            data = self._E.dataframe["weight"]
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
        return len(self._nodes), len(self._edges)

    def __str__(self):
        """
        String representation of hypergraph

        Returns
        -------
        str

        """
        return f"{self.name} <class 'hypernetx.classes.hypergraph.Hypergraph'>"

    def __repr__(self):
        """
        String representation of hypergraph

        Returns
        -------
        str

        """
        return f"{self.name} hypernetx.classes.hypergraph.Hypergraph"

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
        item : hashable

        """
        return item in self._nodes

    def __getitem__(self, node):
        """
        Returns the neighbors of node in the Hypergraph. These
        will be the nodes sharing some edge with the node.

        Parameters
        ----------
        node : hashable
            If hashable, then must be uid of node in hypergraph

        Returns
        -------
        neighbors(node) : iterator

        """
        return self.neighbors(node)

    def clone(self, name=None):
        """
        Create a deep copy of the hypergraph

        Parameters
        ----------
        name : str, optional, default = None

        Returns
        -------
        : Hypergraph
        """
        return self._construct_hyp_from_stores(
            self.incidences.to_dataframe, name=f"{self.name}_clone"
        )

    def __eq__(self, other):
        if type(other) is type(self):
            return self.incidences.items == other.incidences.items
        return False

    def rename(self, edges=None, nodes=None, name=None, inplace=True):
        """
        _summary_

        Parameters
        ----------
        edges : dict, optional, default = None
            dictionary of replacement edge_uids
        nodes : dict, optional, default = None
            dictionary of replacement node_uids
        name : str, optional, default = None

        Returns
        -------
        : Hypergraph
        """
        if edges is None and nodes is None:
            return self
        else:
            edf = self.edges.to_dataframe
            ndf = self.nodes.to_dataframe
            df = self.incidences.to_dataframe
        if edges is not None:
            edf = edf.rename(index=edges)
            df = df.rename(index=edges, level=0)
        if nodes is not None:
            ndf = ndf.rename(index=nodes)
            df = df.rename(index=nodes, level=1)
        eps = PropertyStore(edf)
        nps = PropertyStore(ndf)
        return self._construct_hyp_from_stores(
            df, edge_ps=eps, node_ps=nps, name=name, inplace=inplace
        )

    def get_cell_properties(self, edge_uid, node_uid, prop_name=None):
        """Get cell properties on a specified edge and node

        Parameters
        ----------
        edge_uid : str | int
            edgeid
        node_uid : str | int
            nodeid
        prop_name : str, optional, default = None
            name of a cell property; if None, all cell properties will be returned

        Returns
        -------
        : Any
            cell property value if `prop_name` is provided, otherwise ``dict`` of all
            cell properties and values
        """
        if prop_name is None:
            return self.incidences.property_store.get_properties((edge_uid, node_uid))
        return self.incidences.property_store.get_property(
            (edge_uid, node_uid), prop_name
        )

    def get_properties(self, uid, level=0, prop_name=None):
        """
        Returns an object's specific property or all properties

        Parameters
        ----------
        uid : hashable
            edge or node id
        level : int | None , optional, default = None
            Enter 0 for edges and 1 for nodes.
        prop_name : str | None, optional, default = None
            if None then all properties associated with the object will be returned.

        Returns
        -------
        : Any
            single property or dictionary of properties
        """

        if level == 0:
            store = self._edges
        elif level == 1:
            store = self._nodes
        elif level == 2:
            store = self._E
        if prop_name is None:  ## rewrite for edges and nodes.
            return store.property_store.get_properties(uid)
        return store.property_store.get_property(uid, prop_name)

    def get_linegraph(self, s=1, edges=True):
        """
        Creates an ::term::s-linegraph for the Hypergraph.
        If edges=True (default)then the edges will be the vertices of the line
        graph. Two vertices are connected by an s-line-graph edge if the
        corresponding hypergraph edges intersect in at least s hypergraph nodes.
        If edges=False, the hypergraph nodes will be the vertices of the line
        graph. Two vertices are connected if the nodes they correspond to share
        at least s incident (hyper)edges.

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
        g.add_nodes_from(Amaplst)
        g.add_edges_from(A)
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
        """
        Populate state_dict with default values
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
        Returns the size for each edge.

        Returns
        -------
        list
            a list of sizes of each edge.
        """
        if "edge_size_dist" not in self._state_dict:
            dist = np.array(np.sum(self.incidence_matrix(), axis=0))[0].tolist()
            self.set_state(edge_size_dist=dist)
        return self._state_dict["edge_size_dist"]

    def degree(self, node_uid, s=1, max_size=None):
        """
        The number of edges of size at least s and at most
        max_size that contain the node.

        Parameters
        ----------
        node_uid : hashable
            Identifier for the node.
        s : int, optional, default=1
            The smallest size (must be positive) of an edge to consider in degree.
        max_size : int or None, optional, default=None
            The largest size (must be positive) of edge to consider in degree.

        Returns
        -------
        int
            The number of edges of size at least s and at most
            max_size that contain the node.
        """
        if s == 1 and max_size is None:
            return len(self._nodes.memberships[node_uid])

        ### This could possibly be done more efficiently on the dataframe.
        memberships = set()
        for edge in self._nodes.memberships[node_uid]:
            size = len(self.edges[edge])
            if size >= s and (max_size is None or size <= max_size):
                memberships.add(edge)
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
        node : hashable
            uid for a node in hypergraph

        s : int, list, optional, default = 1
            Minimum number of edges shared by neighbors with node.

        Returns
        -------
        neighbors : list
            s-neighbors share at least s edges in the hypergraph

        """
        if node not in self.nodes:
            warnings.warn(f"{node} is not in hypergraph {self.name}.")
            return []
        if node in self._state_dict["neighbors"][s]:
            return self._state_dict["neighbors"][s][node]

        inc_matrix = self.incidence_matrix()
        rdx = self._state_dict["labels"]["nodes"]
        jdx = np.where(rdx == node)
        idx = (inc_matrix[jdx].dot(inc_matrix.T) >= s) * 1
        idx = np.nonzero(idx)[1]
        neighbors = list(rdx[idx])
        if len(neighbors) > 0:
            neighbors.remove(node)
            self._state_dict["neighbors"][s][node] = neighbors
        else:
            self._state_dict["neighbors"][s][node] = []
        return self._state_dict["neighbors"][s][node]

    def edge_neighbors(self, edge, s=1):
        """
        The edges in hypergraph which share s nodes(s) with edge.

        Parameters
        ----------
        edge : hashable
            uid for a edge in hypergraph

        s : int, list, optional, default = 1
            Minimum number of nodes shared by neighbors edge node.

        Returns
        -------
        list
            a list of edge neighbors

        """

        if edge not in self.edges:
            warnings.warn(f"Edge is not in hypergraph {self.name}.")
            return []
        if edge in self._state_dict["edge_neighbors"][s]:
            return self._state_dict["edge_neighbors"][s][edge]

        inc_matrix = self.incidence_matrix()
        cdx = self._state_dict["labels"]["edges"]
        jdx = np.where(cdx == edge)
        idx = (inc_matrix.T[jdx].dot(inc_matrix) >= s) * 1
        idx = np.nonzero(idx)[1]
        edge_neighbors = list(cdx[idx])
        if len(edge_neighbors) > 0:
            edge_neighbors.remove(edge)
            self._state_dict["edge_neighbors"][s][edge] = edge_neighbors
        else:
            self._state_dict["edge_neighbors"][s][edge] = []
        return self._state_dict["edge_neighbors"][s][edge]

    def adjacency_matrix(self, s=1, index=False):
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
        The unweighted :term:`s-auxiliary matrix` for hypergraph

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

    def bipartite(self, keep_data=False, directed=False):
        """
        Creates a bipartite NetworkX graph from hypergraph.
        The nodes and (hyper)edges of hypergraph become the nodes of bipartite
        graph. For every (hyper)edge e in the hypergraph and node v in e there
        is an edge (e,v) in the graph.

        Parameters
        ----------
        keep_data : bool, optional, default = False
            If True the node and edge data from the hypergraph will be added to
            the NetworkX graph
        directed : bool, optional, default = False
            If True the graph edges will be directed so that the hypergraph
            edges are the sources and the hypergraph nodes are the targets

        Returns
        -------
        networkx.Graph or DiGraph
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
        self, incidence_df, edge_ps=None, node_ps=None, name=None, inplace=False
    ):
        if inplace:
            h = self
            name = self.name
        else:
            h = Hypergraph()

        incidence_store = IncidenceStore(
            pd.DataFrame(incidence_df.index.tolist(), columns=["edges", "nodes"])
        )
        incidence_ps = PropertyStore(
            incidence_df, default_weight=self.incidences.default_weight
        )
        h._E = HypergraphView(incidence_store, 2, incidence_ps)

        if edge_ps is None:
            edge_ps = PropertyStore(
                self.edges.to_dataframe, default_weight=self.edges.default_weight
            )
        h._edges = HypergraphView(incidence_store, 0, edge_ps)

        if node_ps is None:
            node_ps = PropertyStore(
                self.nodes.to_dataframe, default_weight=self.nodes.default_weight
            )
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

        share_properties : bool, optional, default = True
            Whether or not to tie the edge and node properties of
            objects in the dual to objects in the hypergraph.
            If True, a change to edge and node properties in one will
            be reflected in the other.

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

        if share_properties:
            edge_ps = self._nodes.property_store
            node_ps = self._edges.property_store
        else:
            edge_ps = PropertyStore(self.nodes.to_dataframe)
            node_ps = PropertyStore(self.edges.to_dataframe)

        hdual = self._construct_hyp_from_stores(
            dsetsystem,
            edge_ps=edge_ps,
            node_ps=node_ps,
            name=name or str(self.name) + "_dual",
        )
        return hdual

    def equivalence_classes(self, edges=True):
        """
        Returns the equivalence classes created by collapsing edges or nodes.

        Parameters
        ----------
        edges : bool, optional, default=True
            If True collapses edges, otherwise collapses nodes.

        Returns
        -------
        list
            A list of sets of edges or nodes

        See Also
        --------
        collapse_edges
        collapse_nodes
        collapse_nodes_and_edges
        """
        level = 0 if edges else 1
        return self._E.incidence_store.equivalence_classes(level=level)

    def collapse_edges(
        self,
        name=None,
        use_uids=None,
        use_counts=False,
        return_counts=True,
        return_equivalence_classes=False,
        aggregate_edges_by={"weight": "sum"},
        aggregate_cells_by={"weight": "sum"},
    ):
        """
        Returns a new hypergraph by collapsing edges.

        Parameters
        ----------

        name: str, optional, default = None

        use_uids: list, optional, default = None
            Specify the edge identifiers to use as representatives
            for a single equivalence class. If two identifiers occur in the
            same equivalence class, the first one found will be used.

        use_counts: boolean, optional, default = False
            Rename the equivalence class representatives as `<uid>:<size of class>`

        return_counts: bool, optional, default = True
            Add the size of the equivalence class to the properties
            associated to the representative in the collapsed hypergraph using
            keyword: `eclass_size`

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes keyed by a
            representative from each class

        aggregate_edges_by, aggregate_cells_by : dict, optional, default = {'weight':'sum'}
            dictionary of aggregation methods keyed by column names
            in the properties dataframes, does not apply to misc_properties.
            Defaults to 'first' on unlisted columns.
            See pandas.core.groupby.DataFrameGroupBy.agg for usage examples with
            dictionaries

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Collapses the edges of Hypergraph. Two edges are
        duplicates if their respective elements are the same.
        Using this as an equivalence relation, the uids of the edges
        are partitioned into equivalence classes. A single member of the
        equivalence class is chosen to represent the class.

        Example
        -------

            >>> data = {'E1': ('a', 'b'), 'E2': ('a', 'b')}
            >>> h = Hypergraph(data)
            >>> h.incidence_dict
            {'E1': ['a', 'b'], 'E2': ['a', 'b']}
            >>> h.collapse_edges().incidence_dict
            {'E1': ['a'], 'E2': ['a']}
            >>> h.collapse_edges(use_counts=True).incidence_dict
            {'E1: ['a:2'], 'E2': ['a:2']}
            >>> h.collapse_edges().properties.to_dict(orient='records')
            [{'weight': 2, 'misc_properties': {'eclass_size': 2}}]



        """
        _, eclasses = self._E.incidence_store.collapse_identical_elements(
            0, use_keys=use_uids
        )

        ndf = self.edges.to_dataframe
        df = self.incidences.to_dataframe

        if use_uids is not None:  ## then eclasses will reorder the dataframes
            ## so the first occurence of the class is one of these uids.
            newindex = []
            for v in eclasses.values():
                newindex += v
            ndf = ndf.loc[newindex]
            df = df.loc[newindex, :]

        if use_counts:
            mapper = {vv: f"{k}:{len(v)}" for k, v in eclasses.items() for vv in v}
        else:
            mapper = {vv: k for k, v in eclasses.items() for vv in v}

        ndf = ndf.rename(index=mapper)
        ndf = _agg_rows(ndf, ndf.index, aggregate_edges_by)
        edge_ps = PropertyStore(ndf)

        df = df.rename(index=mapper, level=0)
        df = _agg_rows(df, ["edges", "nodes"], aggregate_cells_by)

        node_ps = PropertyStore(self.nodes.to_dataframe)
        H = self._construct_hyp_from_stores(
            df, edge_ps=edge_ps, node_ps=node_ps, name=name
        )

        if return_counts:
            if use_counts:
                for nd in H.edges:
                    H.edges[nd].equivalence_class_size = int(nd.split(":")[1])
            else:
                for nd in H.edges:
                    H.edges[nd].equivalence_class_size = len(eclasses[nd])

        if return_equivalence_classes:
            return H, {mapper[k]: eclasses[k] for k in eclasses}
        else:
            return H

    def collapse_nodes(
        self,
        name=None,
        use_uids=None,
        use_counts=False,
        return_counts=True,
        return_equivalence_classes=False,
        aggregate_nodes_by={"weight": "sum"},
        aggregate_cells_by={"weight": "sum"},
    ):
        """
        Returns a new hypergraph by collapsing nodes.

        Parameters
        ----------

        name: str, optional, default = None

        use_uids: list, optional, default = None
            Specify the node identifiers to use as representatives
            for a single equivalence class. If two identifiers occur in the
            same equivalence class, the first one found will be used.

        use_counts: boolean, optional, default = False
            Rename the equivalence class representatives as `<uid>:<size of class>`

        return_counts: bool, optional, default = True
            Add the size of the equivalence class to the properties
            associated to the representative in the collapsed hypergraph using
            keyword: `eclass_size`

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes keyed by a
            representative from each class

        aggregate_nodes_by, aggregate_cells_by : dict, optional, default = {'weight':'sum'}
            dictionary of aggregation methods keyed by column names
            in the properties dataframes, does not apply to misc_properties.
            Defaults to 'first' on unlisted columns.
            See pandas.core.groupby.DataFrameGroupBy.agg for usage examples with
            dictionaries

        Returns
        -------
        new hypergraph : Hypergraph

        Notes
        -----
        Collapses the nodes of Hypergraph. Two nodes are
        duplicates if their respective memberships are the same.
        Using this as an equivalence relation, the uids of the nodes
        are partitioned into equivalence classes. A single member of the
        equivalence class is chosen to represent the class.

        Example
        -------

            >>> data = {'E1': ('a', 'b'), 'E2': ('a', 'b')}
            >>> h = Hypergraph(data)
            >>> h.incidence_dict
            {'E1': ['a', 'b'], 'E2': ['a', 'b']}
            >>> h.collapse_nodes().incidence_dict
            {'E1': ['a'], 'E2': ['a']}
            >>> h.collapse_nodes(use_counts=True).incidence_dict
            {'E1: ['a:2'], 'E2': ['a:2']}
            >>> h.collapse_nodes().properties.to_dict(orient='records')
            [{'weight': 2, 'misc_properties': {'eclass_size': 2}}]
        """
        _, eclasses = self._E.incidence_store.collapse_identical_elements(
            1, use_keys=use_uids
        )

        ndf = self.nodes.to_dataframe
        df = self.incidences.to_dataframe

        if use_uids is not None:  ## then eclasses will reorder the dataframes
            ## so the first occurence of the class is one of these uids.
            newindex = []
            for v in eclasses.values():
                newindex += v
            ndf = ndf.loc[newindex]
            df = df.swaplevel().loc[newindex, :].swaplevel()

        if use_counts:
            mapper = {vv: f"{k}:{len(v)}" for k, v in eclasses.items() for vv in v}
        else:
            mapper = {vv: k for k, v in eclasses.items() for vv in v}

        ndf = ndf.rename(index=mapper)
        ndf = _agg_rows(ndf, ndf.index, aggregate_nodes_by)
        node_ps = PropertyStore(ndf)

        df = df.rename(index=mapper, level=1)
        df = _agg_rows(df, ["edges", "nodes"], aggregate_cells_by)

        edge_ps = PropertyStore(self.edges.to_dataframe)
        H = self._construct_hyp_from_stores(
            df, edge_ps=edge_ps, node_ps=node_ps, name=name
        )

        if return_counts:
            if use_counts:
                for nd in H.nodes:
                    H.nodes[nd].equivalence_class_size = int(nd.split(":")[1])
            else:
                for nd in H.nodes:
                    H.nodes[nd].equivalence_class_size = len(eclasses[nd])

        if return_equivalence_classes:
            return H, {mapper[k]: eclasses[k] for k in eclasses}
        else:
            return H

    def collapse_nodes_and_edges(
        self,
        name=None,
        use_edge_uids=None,
        use_node_uids=None,
        use_counts=False,
        return_counts=True,
        return_equivalence_classes=False,
        aggregate_nodes_by={"weight": "sum"},
        aggregate_edges_by={"weight": "sum"},
        aggregate_cells_by={"weight": "sum"},
    ):
        """
        Returns a new hypergraph by collapsing nodes and edges.

        Parameters
        ----------

        name: str, optional, default = None

        return_equivalence_classes: boolean, optional, default = False
            Returns a dictionary of edge equivalence classes keyed by a
            representative from each class

        use_edge_uids, use_node_uids: list, optional, default = None
            Specify the edge and node identifiers to use as representatives
            for a single equivalence class. If two identifiers occur in the
            same equivalence class, the first one found will be used.

        use_counts: boolean, optional, default = False
            Rename the equivalence class representatives as `<uid>:<size of class>`

        return_counts: bool, optional, default = True
            Add the size of the equivalence class to the properties
            associated to the representative in the collapsed hypergraph using
            keyword: `eclass_size`

        Returns
        -------
        new hypergraph : Hypergraph
        node equivalence classes : dict
        edge equivalence classes : dict

        Notes
        -----
        Collapses the Nodes and Edges of Hypergraph. Two nodes(edges) are
        duplicates if their respective memberships(elements) are the same.
        Using this as an equivalence relation, the uids of the nodes(edges)
        are partitioned into equivalence classes. A single member of the
        equivalence class is chosen to represent the class.

        Example
        -------

            >>> data = {'E1': ('a', 'b'), 'E2': ('a', 'b')}
            >>> h = Hypergraph(data)
            >>> h.incidence_dict
            {'E1': ['a', 'b'], 'E2': ['a', 'b']}
            >>> h.collapse_nodes_and_edges().incidence_dict
            {'E1': ['a']}
            >>> h.collapse_nodes_and_edges(use_counts=True).incidence_dict
            {'E1:2': ['a:2']}
        """

        if return_equivalence_classes:
            temp, neq = self.collapse_nodes(
                return_equivalence_classes=True,
                use_uids=use_node_uids,
                use_counts=use_counts,
                return_counts=return_counts,
                aggregate_nodes_by=aggregate_nodes_by,
                aggregate_cells_by=aggregate_cells_by,
            )
            ntemp, eeq = temp.collapse_edges(
                name=name,
                return_equivalence_classes=True,
                use_uids=use_edge_uids,
                use_counts=use_counts,
                return_counts=return_counts,
                aggregate_edges_by=aggregate_edges_by,
                aggregate_cells_by=aggregate_cells_by,
            )
            return ntemp, neq, eeq
        else:
            temp = self.collapse_nodes(
                use_uids=use_node_uids,
                use_counts=use_counts,
                return_counts=return_counts,
                aggregate_nodes_by=aggregate_nodes_by,
                aggregate_cells_by=aggregate_cells_by,
            )
            return temp.collapse_edges(
                name=name,
                use_uids=use_edge_uids,
                use_counts=use_counts,
                return_counts=return_counts,
                aggregate_edges_by=aggregate_edges_by,
                aggregate_cells_by=aggregate_cells_by,
            )

    def restrict_to_nodes(self, nodes, name=None):
        """
        New hypergraph gotten by restricting to nodes

        Parameters
        ----------
        nodes : Iterable
            node identifiers to restrict to
        name : str | int, optional
            node identifier, by default None

        Returns
        -------
            : Hypergraph
        """

        keys = list(set(self._state_dict["labels"]["nodes"]).difference(nodes))
        return self._remove(keys, level=1, name=name, inplace=False)

    def restrict_to_edges(self, edges, name=None):
        """New hypergraph gotten by restricting to edges

        Parameters
        ----------
        edges : Iterable
            edgeids to restrict to
        name : str | int, optional
            edge identifier, by default None

        Returns
        -------
        hnx.Hypergraph

        """
        keys = list(set(self._state_dict["labels"]["edges"]).difference(edges))
        return self._remove(keys, level=0, name=name, inplace=False)

    def add_edge(self, uid, inplace=True, **attr):
        """
        Add a single edge with attributes to edge properties.
        Does not add an incidence to the hypergraph.

        Parameters
        ----------
        uid : int | str
            edge_uid
        **attr : key = value pairs to be properties

        Returns
        -------
        Hypergraph
        """
        return self._add_items_from([(uid, attr)], 0, inplace=inplace)

    def add_edges_from(self, edges, inplace=True):
        """Edges must be a list of uids and/or tuples
        of the form (uid,data) where data is dictionary"""
        newedges = self._process_items(edges)
        return self._add_items_from(newedges, 0, inplace=inplace)

    def add_node(self, uid, inplace=True, **attr):
        return self._add_items_from([(uid, attr)], 1, inplace=inplace)

    def add_nodes_from(self, nodes, inplace=True):
        """Nodes must be a list of uids and/or tuples
        of the form (uid,data) where data is dictionary"""
        newnodes = self._process_items(nodes)
        return self._add_items_from(newnodes, 1, inplace=inplace)

    def _process_items(self, items):
        new_items = list()
        for item in items:
            if not isinstance(item, tuple):
                new_items.append((item, {}))
            else:
                new_items.append(item)
        return new_items

    def add_nodes_to_edges(self, edge_dict, inplace=True):
        """
        Parameters
        ----------
        edge_dict: dict[str, list[str | int] | dict[str, dict]]
            The edge dictionary must be a dictionary of edges as the keys and a list of nodes or a dictionary
            of nodes to properties as the values.

        Returns
        -------
        Hypergraph
            A new Hypergraph with the updated edges and their newly added nodes
        """
        items = list()
        for ed, nodes in edge_dict.items():
            if isinstance(nodes, dict):
                for nd, data in nodes.items():
                    items.append(((ed, nd), data))
            else:
                for nd in nodes:
                    items.append(((ed, nd), {}))
        return self._add_items_from(items, 2, inplace=inplace)

    def add_incidence(self, edge_id, node_id, inplace=True, **attr):
        return self._add_items_from([((edge_id, node_id), attr)], 2, inplace=inplace)

    def add_incidences_from(self, incidences, inplace=True):
        """
        Adds incidences to Hypergraph

        Parameters
        ----------
        incidences: list[str | int, str | int], list[tuple[str | int, str | int, dict[str, Any]]
            Incidence pairs must be a list of uids of the form (edge_uid,node_uid)
            and/or tuples of the form (edge_uid, node_uid,data) where data is a
            dictionary.

        Returns
        -------
        Hypergraph
            A new Hypergraph with incidences added.
        """
        newincidences = list()
        for pr in incidences:
            if len(pr) == 2:
                newincidences.append((pr, {}))
            else:
                newincidences.append(((pr[0], pr[1]), pr[2]))
        return self._add_items_from(newincidences, 2, inplace=inplace)

    def _add_items_from(self, items, level, inplace=True):
        """Items must be a list of tuples
        of the form (uid,data) where data is dictionary"""
        df = self.incidences._property_store
        ep = self.edges._property_store
        ndp = self.nodes._property_store
        hv = [ep, ndp, df][level]
        for item in items:
            uid = item[0]
            data = item[1]
            hv.set_properties(uid, data)
        return self._construct_hyp_from_stores(
            df.properties, edge_ps=ep, node_ps=ndp, name=self.name, inplace=inplace
        )

    #### This should follow behavior of restrictions
    def remove_edges(self, keys, name=None):
        if isinstance(keys, list):
            return self._remove(keys, level=0, name=name)
        else:
            return self._remove([keys], level=0, name=name)

    def remove_nodes(self, keys, name=None):
        if isinstance(keys, list):
            return self._remove(keys, level=1, name=name)
        else:
            return self._remove([keys], level=1, name=name)

    def remove_incidences(self, keys, name=None):
        if isinstance(keys, list):
            return self._remove(keys, name=name)
        else:
            return self._remove([keys], name=name)

    def _remove(
        self,
        uid_list,
        level=2,
        name=None,
        inplace=False,
    ):
        """Creates a new hypergraph with nodes and/or edges indexed by keys
        removed. More efficient for creating a restricted hypergraph if the
        restricted set is greater than what is being removed.

        Parameters
        ----------
        uid_list : list
            list of uids from edges, nodes, or incidence pairs(listed as tuples)
        level : int, optional, default = 2
            Enter 0 to remove edges.
            Enter 1 to remove nodes.
            Enter 2 to remove incidence pairs as tuples
        name : str, optional, default = None
            Name of new hypergraph
        inplace : bool, default = False
            Whether to replace the current hypergraph with the new one.

        Returns
        -------
        : hnx.Hypergraph

        Notes
        -----
        Removal of a node or edge from the hypergraph will remove all
        instances of these objects from incidence pairs and from the data.
        Removal of an incidence pair, only removes that pair but does not
        affect the user data attached to the edge and node in the pair.

        """
        if inplace:
            df = self.incidences.properties
            ep = self.edges.property_store
            ndp = self.nodes.property_store
        else:
            df = self.incidences.to_dataframe
            ep = self.edges.property_store.copy(deep=True)
            ndp = self.nodes.property_store.copy(deep=True)
        if level in [0, 1]:
            hv = [ep, ndp][level]
            df = df.drop(labels=uid_list, level=level, errors="ignore")
            hv._data = hv._data.drop(labels=uid_list, errors="ignore")
        else:
            df = df.drop(labels=uid_list, errors="ignore")
        return self._construct_hyp_from_stores(
            df, edge_ps=ep, node_ps=ndp, name=self.name, inplace=inplace
        )

    def toplexes(self, return_hyp=False, name=None):
        """
        Computes a maximal collection of toplexes for the hypergraph.
        A toplex is a hyperedge, which is not contained in any other
        hyperedge. If return_hyp is set to True then
        returns the :term:`simple hypergraph` gotten by restricting
        to the toplexes.

        Parameters
        ----------
        return_hyp: bool, optional, default = False
        name: str, optional, default = None
        """

        def operate(a, b):
            return np.mod(np.mod(a * b, 2) + b, 2)

        df = self.incidence_dataframe().T
        toplexes = []
        while True:
            edge_sizes = dict(df.sum(axis=1))
            edges = np.array(
                sorted(df.index, key=lambda x: edge_sizes[x], reverse=True)
            )
            toplexes.append(edges[0])
            df = df.loc[edges]
            mat = df.values
            df = df.loc[edges[operate(mat[0], mat).sum(axis=1).nonzero()]]
            if len(df.index) == 0:
                break
        if return_hyp == True:
            return self.restrict_to_edges(toplexes)
        else:
            return toplexes

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
        return self.remove_edges(singletons, name=name)

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
            If True, return edge components; otherwise, return node components

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
        return self.s_connected_components(s=1, edges=edges, return_singletons=True)

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
        Static method creates a Hypergraph from a NetworkX bipartite graph.
        Still to come: capturing edge and node properties from the graph for use
        in the hypergraph.

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
            >>> B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
            >>> H = Hypergraph.from_bipartite(B, nodes=1)
            >>> list(H.nodes), list(H.edges)
            # output: (['a', 'b', 'c'], [1, 2, 3, 4])
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
        df = pd.DataFrame({"edges": edges, "nodes": nodes, "weights": weights})
        return Hypergraph(df, cell_weight_col="weights", name=name, **kwargs)

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
        name=None,
        fillna=0,
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

        name : (optional) string, default = None

        fillna : float, default = 0
            a real value to place in empty cell, all-zero columns will not
            generate an edge.

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

        df = df.fillna(fillna)

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
        Hypergraph obtained by joining incidences from self and other.
        Removes duplicates and uses properties of self.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
        Hypergraph

        """
        incidence_df = self._combine_properties_dataframes(
            self.incidences.to_dataframe, other.incidences.to_dataframe
        )
        edges_data = self._combine_properties_dataframes(
            self.edges.to_dataframe, other.edges.to_dataframe
        )
        nodes_data = self._combine_properties_dataframes(
            self.nodes.to_dataframe, other.nodes.to_dataframe
        )

        return self._construct_hyp_from_stores(
            incidence_df,
            edge_ps=PropertyStore(edges_data),
            node_ps=PropertyStore(nodes_data),
        )

    def _combine_properties_dataframes(self, df1, df2):
        df = pd.concat([df1, df2])
        return df[~df.index.duplicated(keep="first")]

    def difference(self, other, name=None):
        """
        Hypergraph obtained by restricting to incidences in self but not in other.

        Parameters
        ----------
        other : Hypergraph

        Returns
        -------
         : Hypergraph

        """
        ndx = list(self.incidences.items.difference(other.incidences.items))
        ndf = self.incidences.to_dataframe.loc[ndx]
        return self._construct_hyp_from_stores(ndf, name=name)

    def intersection(self, other, name=None):
        """
        The hypergraph gotten by restricting to incidence pairs contained in
            both self and other. Properties inherited from self.

        Parameters
        ----------
        other : Hypergraph
        name : str, optional
             by default None

        Returns
        -------
        : Hypergraph

        """
        ndx = list(self.incidences.items.intersection(other.incidences.items))
        ndf = self.incidences.to_dataframe.loc[ndx]
        return self._construct_hyp_from_stores(ndf, name=name)

    def union(self, other, name=None):
        """
        The hypergraph gotten by joining incidence pairs contained in
            self and other. Duplicates removed. Properties inherited from self.
            Same as sum.

        Parameters
        ----------
        other : Hypergraph
        name : str, optional
             by default None

        Returns
        -------
        : Hypergraph

        """
        return self.sum(other)


def _agg_rows(df, groupby, rule_dict=None):
    """
    Helper method for collapsing nodes and edges in hypergraph

    Parameters
    ----------
    df : pandas.DataFrame
    groupby : index or columns aggregating on
    rule_dict : dict, optional, defaults as 'first' for all keys
        dictionary keyed by df columns where values are the
        aggregation rules e.g. 'first', 'sum', 'last'

    Returns
    -------
    : pandas.DataFrame
    """
    default_agg = {col: "first" for col in df.columns}
    for k, v in rule_dict.items():
        if k in default_agg:
            default_agg[k] = v
    return df.reset_index().groupby(groupby).agg(default_agg)
