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

from hypernetx.classes import Entity, EntitySet
from hypernetx.exception import HyperNetXError
from hypernetx.utils.decorators import warn_nwhy
from hypernetx.classes.helpers import merge_nested_dicts, dict_depth

__all__ = ["Hypergraph"]

T = TypeVar("T", bound=Union[str, int])


class Hypergraph:
    """
    Parameters
    ----------

    setsystem : (optional) dict of iterables, dict of dicts,iterable of iterables,
        pandas.DataFrame, numpy.ndarray, default = None
        See SetSystem above for additional setsystem requirements.

    edge_col : (optional) str | int, default = 0
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for (hyper)edge ids. Will be used to reference edgeids for
        all set systems.

    node_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for node ids. Will be used to reference nodeids for all set systems.

    cell_weight_col : (optional) str | int, default = None
        column index (or name) in pandas.dataframe or numpy.ndarray used for
        referencing cell weights. For a dict of dicts references key in cell
        property dicts.

    cell_weights : (optional) Sequence[float,int] | int |  float , default = 1.0
        User specified cell_weights or default cell weight.
        Sequential values are only used if setsystem is a
        dataframe or ndarray in which case the sequence must
        have the same length and order as these objects.
        Sequential values are ignored for dataframes if cell_weight_col is already
        a column in the data frame.
        If cell_weights is assigned a single value
        then it will be used as default for missing values or when no cell_weight_col
        is given.

    cell_properties : (optional) Sequence[int | str] | Mapping[T,Mapping[T,Mapping[str,Any]]],
        default = None
        Column names from pd.DataFrame to use as cell properties
        or a dict assigning cell_property to incidence pairs of edges and
        nodes. Will generate a misc_cell_properties, which may have variable lengths per cell.

    misc_cell_properties : (optional) str | int, default = None
        Column name of dataframe corresponding to a column of variable
        length property dictionaries for the cell. Ignored for other setsystem
        types.

    aggregateby : (optional) str, dict, default = 'first'
        By default duplicate edge,node incidences will be dropped unless
        specified with `aggregateby`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information.

    edge_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with edge ids.
        First column of dataframe or keys of dict link to edge ids in
        setsystem.

    node_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with node ids.
        First column of dataframe or keys of dict link to node ids in
        setsystem.

    properties : (optional) pd.DataFrame | dict, default = None
        Concatenation/union of edge_properties and node_properties.
        By default, the object id is used and should be the first column of
        the dataframe, or key in the dict. If there are nodes and edges
        with the same ids and different properties then use the edge_properties
        and node_properties keywords.

    misc_properties : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    edge_weight_prop : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    node_weight_prop : (optional) str, default = None,
        Name of property in node_properties to use for weight.

    weight_prop : (optional) str, default = None
        Name of property in properties to use for 'weight'

    default_edge_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    default_node_weight : (optional) int | float, default = 1
        Used when node weight property is missing or undefined

    name : (optional) str, default = None
        Name assigned to hypergraph


    ======================
    Hypergraphs in HNX 2.0
    ======================

    An hnx.Hypergraph H = (V,E) references a pair of disjoint sets:
    V = nodes (vertices) and E = (hyper)edges.

    HNX allows for multi-edges by distinguishing edges by
    their identifiers instead of their contents. For example, if
    V = {1,2,3} and E = {e1,e2,e3},
    where e1 = {1,2}, e2 = {1,2}, and e3 = {1,2,3},
    the edges e1 and e2 contain the same set of nodes and yet
    are distinct and are distinguishable within H = (V,E).

    New as of version 2.0, HNX provides methods to easily store and
    access additional metadata such as cell, edge, and node weights.
    Metadata associated with (edge,node) incidences
    are referenced as **cell_properties**.
    Metadata associated with a single edge or node is referenced
    as its **properties**.

    The fundamental object needed to create a hypergraph is a **setsystem**. The
    setsystem defines the many-to-many relationships between edges and nodes in
    the hypergraph. Cell properties for the incidence pairs can be defined within
    the setsystem or in a separate pandas.Dataframe or dict.
    Edge and node properties are defined with a pandas.DataFrame or dict.

    SetSystems
    ----------
    There are five types of setsystems currently accepted by the library.

    1.  **iterable of iterables** : Barebones hypergraph uses Pandas default
        indexing to generate hyperedge ids. Elements must be hashable.: ::

        >>> H = Hypergraph([{1,2},{1,2},{1,2,3}])

    2.  **dictionary of iterables** : the most basic way to express many-to-many
        relationships providing edge ids. The elements of the iterables must be
        hashable): ::

        >>> H = Hypergraph({'e1':[1,2],'e2':[1,2],'e3':[1,2,3]})

    3.  **dictionary of dictionaries**  : allows cell properties to be assigned
        to a specific (edge, node) incidence. This is particularly useful when
        there are variable length dictionaries assigned to each pair: ::

        >>> d = {'e1':{ 1: {'w':0.5, 'name': 'related_to'},
        >>>             2: {'w':0.1, 'name': 'related_to',
        >>>                 'startdate': '05.13.2020'}},
        >>>      'e2':{ 1: {'w':0.52, 'name': 'owned_by'},
        >>>             2: {'w':0.2}},
        >>>      'e3':{ 1: {'w':0.5, 'name': 'related_to'},
        >>>             2: {'w':0.2, 'name': 'owner_of'},
        >>>             3: {'w':1, 'type': 'relationship'}}

        >>> H = Hypergraph(d, cell_weight_col='w')

    4.  **pandas.DataFrame** For large datasets and for datasets with cell
        properties it is most efficient to construct a hypergraph directly from
        a pandas.DataFrame. Incidence pairs are in the first two columns.
        Cell properties shared by all incidence pairs can be placed in their own
        column of the dataframe. Variable length dictionaries of cell properties
        particular to only some of the incidence pairs may be placed in a single
        column of the dataframe. Representing the data above as a dataframe df:

        +-----------+-----------+-----------+-----------------------------------+
        |   col1    |   col2    |   w       |  col3                             |
        +-----------+-----------+-----------+-----------------------------------+
        |   e1      |   1       |   0.5     | {'name':'related_to'}             |
        +-----------+-----------+-----------+-----------------------------------+
        |   e1      |   2       |   0.1     | {"name":"related_to",             |
        |           |           |           |  "startdate":"05.13.2020"}        |
        +-----------+-----------+-----------+-----------------------------------+
        |   e2      |   1       |   0.52    | {"name":"owned_by"}               |
        +-----------+-----------+-----------+-----------------------------------+
        |   e2      |   2       |   0.2     |                                   |
        +-----------+-----------+-----------+-----------------------------------+
        |   ...     |   ...     |   ...     | {...}                             |
        +-----------+-----------+-----------+-----------------------------------+

        The first row of the dataframe is used to reference each column. ::

        >>> H = Hypergraph(df,edge_col="col1",node_col="col2",
        >>>                 cell_weight_col="w",misc_cell_properties="col3")

    5.  **numpy.ndarray** For homogeneous datasets given in an ndarray a
        pandas dataframe is generated and column names are added from the
        edge_col and node_col arguments. Cell properties containing multiple data
        types are added with a separate dataframe or dict and passed through the
        cell_properties keyword. ::

        >>> arr = np.array([['e1','1'],['e1','2'],
        >>>                 ['e2','1'],['e2','2'],
        >>>                 ['e3','1'],['e3','2'],['e3','3']])
        >>> H = hnx.Hypergraph(arr, column_names=['col1','col2'])


    Edge and Node Properties
    ------------------------
    Properties specific to a single edge or node are passed through the
    keywords: **edge_properties, node_properties, properties**.
    Properties may be passed as dataframes or dicts.
    The first column or index of the dataframe or keys of the dict keys
    correspond to the edge and/or node identifiers.
    If identifiers are shared among edges and nodes, or are distinct
    for edges and nodes, properties may be combined into a single
    object and passed to the **properties** keyword. For example:

    +-----------+-----------+---------------------------------------+
    |   id      |   weight  |   properties                          |
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

        dp = {id1 : {prop1:val1, prop2,val2,...}, id2 : ... }

    A properties dataframe may be used for nodes and edges sharing ids
    but differing in cell properties by adding a level index using 0
    for edges and 1 for nodes:

    +-----------+-----------+-----------+---------------------------+
    |  level    |   id      |   weight  |       properties          |
    +-----------+-----------+-----------+---------------------------+
    |   0       |   e1      |   5.0     |   {'type':'event'}        |
    +-----------+-----------+-----------+---------------------------+
    |   0       |   e2      |    0.52   |   {"name":"owned_by"}     |
    +-----------+-----------+-----------+---------------------------+
    |   ...     |   ...     |    ...    |          {...}            |
    +-----------+-----------+-----------+---------------------------+
    |   1       |   1.2     |   {'color':'red'}                     |
    +-----------+-----------+-----------+---------------------------+
    |   2       |   .003    |   {'name':'Fido','color':'brown'}     |
    +-----------+-----------+-----------+---------------------------+
    |   ...     |   ...     |    ...    |          {...}            |
    +-----------+-----------+-----------+---------------------------+



    Weights
    -------
    The default key for cell and object weights is "weight". The default value
    is 1. Weights may be assigned and/or a new default prescribed in the
    constructor using **cell_weight_col** and **cell_weights** for incidence pairs,
    and using **edge_weight_prop, node_weight_prop, weight_prop,
    default_edge_weight,** and **default_node_weight** for node and edge weights.

    """

    @warn_nwhy
    def __init__(
        self,
        setsystem: Optional[
            pd.DataFrame
            | np.ndarray
            | Mapping[T, Iterable[T]]
            | Iterable[Iterable[T]]
            | Mapping[T, Mapping[T, Mapping[str, Any]]]
        ] = None,
        edge_col: str | int = 0,
        node_col: str | int = 1,
        cell_weight_col: Optional[str | int] = "cell_weights",
        cell_weights: Sequence[float] | float = 1.0,
        cell_properties: Optional[
            Sequence[str | int] | Mapping[T, Mapping[T, Mapping[str, Any]]]
        ] = None,
        misc_cell_properties_col: Optional[str | int] = None,
        aggregateby: str | dict[str, str] = "first",
        edge_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        node_properties: Optional[pd.DataFrame | dict[T, dict[Any, Any]]] = None,
        properties: Optional[
            pd.DataFrame | dict[T, dict[Any, Any]] | dict[T, dict[T, dict[Any, Any]]]
        ] = None,
        misc_properties_col: Optional[str | int] = None,
        edge_weight_prop_col: str | int = "weight",
        node_weight_prop_col: str | int = "weight",
        weight_prop_col: str | int = "weight",
        default_edge_weight: Optional[float | None] = None,
        default_node_weight: Optional[float | None] = None,
        default_weight: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.name = name or ""
        self.misc_cell_properties_col = misc_cell_properties = (
            misc_cell_properties_col or "cell_properties"
        )
        self.misc_properties_col = misc_properties_col = (
            misc_properties_col or "properties"
        )
        self.default_edge_weight = default_edge_weight = (
            default_edge_weight or default_weight
        )
        self.default_node_weight = default_node_weight = (
            default_node_weight or default_weight
        )
        ### cell properties

        if setsystem is None:  #### Empty Case
            self._edges = EntitySet({})
            self._nodes = EntitySet({})
            self._state_dict = {}

        else:  #### DataFrame case
            if isinstance(setsystem, pd.DataFrame):
                if isinstance(edge_col, int):
                    self._edge_col = edge_col = setsystem.columns[edge_col]
                    if isinstance(edge_col, int):
                        setsystem = setsystem.rename(columns={edge_col: "edges"})
                        self._edge_col = edge_col = "edges"
                else:
                    self._edge_col = edge_col

                if isinstance(node_col, int):
                    self._node_col = node_col = setsystem.columns[node_col]
                    if isinstance(node_col, int):
                        setsystem = setsystem.rename(columns={node_col: "nodes"})
                        self._node_col = node_col = "nodes"
                else:
                    self._node_col = node_col

                entity = setsystem.copy()

                if isinstance(cell_weight_col, int):
                    self._cell_weight_col = setsystem.columns[cell_weight_col]
                else:
                    self._cell_weight_col = cell_weight_col

                if cell_weight_col in entity:
                    entity = entity.fillna({cell_weight_col: cell_weights})
                else:
                    entity[cell_weight_col] = cell_weights

                if isinstance(cell_properties, Sequence):
                    cell_properties = [
                        c
                        for c in cell_properties
                        if not c in [edge_col, node_col, cell_weight_col]
                    ]
                    cols = [edge_col, node_col, cell_weight_col] + cell_properties
                    entity = entity[cols]
                elif isinstance(cell_properties, dict):
                    cp = []
                    for idx in entity.index:
                        edge, node = entity.iloc[idx][[edge_col, node_col]].values
                        cp.append(cell_properties[edge][node])
                    entity["cell_properties"] = cp

            else:  ### Cases Other than DataFrame
                self._edge_col = edge_col = edge_col or "edges"
                if node_col == 1:
                    self._node_col = node_col = "nodes"
                else:
                    self._node_col = node_col
                self._cell_weight_col = cell_weight_col

                if isinstance(setsystem, np.ndarray):
                    if setsystem.shape[1] != 2:
                        raise HyperNetXError("Numpy array must have exactly 2 columns.")
                    entity = pd.DataFrame(setsystem, columns=[edge_col, node_col])
                    entity[cell_weight_col] = cell_weights

                elif isinstance(setsystem, dict):
                    ## check if it is a dict of iterables or a nested dict. if the latter then pull
                    ## out the nested dicts as cell properties.
                    ## cell properties must be of the same type as setsystem

                    entity = pd.Series(setsystem).explode()
                    entity = pd.DataFrame(
                        {edge_col: entity.index.to_list(), node_col: entity.values}
                    )

                    if dict_depth(setsystem) > 2:
                        cell_props = dict(setsystem)
                        if isinstance(cell_properties, dict):
                            ## if setsystem is a dict then cell properties must be a dict
                            cell_properties = merge_nested_dicts(
                                cell_props, cell_properties
                            )
                        else:
                            cell_properties = cell_props

                        df = setsystem
                        cp = []
                        wt = []
                        for idx in entity.index:
                            edge, node = entity.values[idx][[0, 1]]
                            wt.append(df[edge][node].get(cell_weight_col, cell_weights))
                            cp.append(df[edge][node])
                        entity[self._cell_weight_col] = wt
                        entity["cell_properties"] = cp

                    else:
                        entity[self._cell_weight_col] = cell_weights

                elif isinstance(setsystem, Iterable):
                    entity = pd.Series(setsystem).explode()
                    entity = pd.DataFrame(
                        {edge_col: entity.index.to_list(), node_col: entity.values}
                    )
                    entity["cell_weights"] = cell_weights

                else:
                    raise HyperNetXError(
                        "setsystem is not supported or is in the wrong format."
                    )

            def props2dict(df=None):
                if df is None:
                    return {}
                elif isinstance(df, pd.DataFrame):
                    return df.set_index(df.columns[0]).to_dict(orient="index")
                else:
                    return dict(df)

            if properties is None:
                if edge_properties is not None or node_properties is not None:
                    if edge_properties is not None:
                        edge_properties = props2dict(edge_properties)
                        for e in entity[edge_col].unique():
                            if not e in edge_properties:
                                edge_properties[e] = {}
                        for v in edge_properties.values():
                            v.setdefault(edge_weight_prop_col, default_edge_weight)
                    else:
                        edge_properties = {}
                    if node_properties is not None:
                        node_properties = props2dict(node_properties)
                        for nd in entity[node_col].unique():
                            if not nd in node_properties:
                                node_properties[nd] = {}
                        for v in node_properties.values():
                            v.setdefault(node_weight_prop_col, default_node_weight)
                    else:
                        node_properties = {}
                    properties = {0: edge_properties, 1: node_properties}
            else:
                if isinstance(properties, pd.DataFrame):
                    if weight_prop_col in properties.columns:
                        properties = properties.fillna(
                            {weight_prop_col: default_weight}
                        )
                    elif misc_properties_col in properties.columns:
                        for idx in properties.index:
                            if not isinstance(
                                properties[misc_properties_col][idx], dict
                            ):
                                properties[misc_properties_col][idx] = {
                                    weight_prop_col: default_weight
                                }
                            else:
                                properties[misc_properties_col][idx].setdefault(
                                    weight_prop_col, default_weight
                                )
                    else:
                        properties[weight_prop_col] = default_weight
                if isinstance(properties, dict):
                    if dict_depth(properties) <= 2:
                        properties = pd.DataFrame(
                            [
                                {"id": k, misc_properties_col: v}
                                for k, v in properties.items()
                            ]
                        )
                        for idx in properties.index:
                            if isinstance(properties[misc_properties_col][idx], dict):
                                properties[misc_properties_col][idx].setdefault(
                                    weight_prop_col, default_weight
                                )
                            else:
                                properties[misc_properties_col][idx] = {
                                    weight_prop_col: default_weight
                                }
                    elif set(properties.keys()) == {0, 1}:
                        edge_properties = properties[0]
                        for e in entity[edge_col].unique():
                            if not e in edge_properties:
                                edge_properties[e] = {
                                    edge_weight_prop_col: default_edge_weight
                                }
                            else:
                                edge_properties[e].setdefault(
                                    edge_weight_prop_col, default_edge_weight
                                )
                        node_properties = properties[1]
                        for nd in entity[node_col].unique():
                            if not nd in node_properties:
                                node_properties[nd] = {
                                    node_weight_prop_col: default_node_weight
                                }
                            else:
                                node_properties[nd].setdefault(
                                    node_weight_prop_col, default_node_weight
                                )
                        for idx in properties.index:
                            if not isinstance(
                                properties[misc_properties_col][idx], dict
                            ):
                                properties[misc_properties_col][idx] = {
                                    weight_prop_col: default_weight
                                }
                            else:
                                properties[misc_properties_col][idx].setdefault(
                                    weight_prop_col, default_weight
                                )

            self.E = EntitySet(
                entity=entity,
                level1=edge_col,
                level2=node_col,
                weight_col=cell_weight_col,
                weights=cell_weights,
                cell_properties=cell_properties,
                misc_cell_props_col=misc_cell_properties_col or "cell_properties",
                aggregateby=aggregateby or "sum",
                properties=properties,
                misc_props_col=misc_properties_col,
            )

            self._edges = self.E
            self._nodes = self.E.restrict_to_levels([1])
            self._dataframe = self.E.cell_properties.reset_index()
            self._data_cols = data_cols = [self._edge_col, self._node_col]
            self._dataframe[data_cols] = self._dataframe[data_cols].astype("category")

            self.__dict__.update(locals())
            self._set_default_state()

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
        return self._dataframe

    @property
    def properties(self):
        """Returns dataframe of edge and node properties.

        Returns
        -------
        pd.DataFrame
        """
        return self.E.properties

    @property
    def edge_props(self):
        """Dataframe of edge properties
        indexed on edge ids

        Returns
        -------
        pd.DataFrame
        """
        return self.E.properties.loc[0]

    @property
    def node_props(self):
        """Dataframe of node properties
        indexed on node ids

        Returns
        -------
        pd.DataFrame
        """
        return self.E.properties.loc[1]

    @property
    def incidence_dict(self):
        """
        Dictionary keyed by edge uids with values the uids of nodes in each
        edge

        Returns
        -------
        dict

        """
        return self.E.incidence_dict

    @property
    def shape(self):
        """
        (number of nodes, number of edges)

        Returns
        -------
        tuple

        """
        return len(self._nodes.elements), len(self._edges.elements)

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
        return iter(self.nodes)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.nodes

        Parameters
        ----------
        item : hashable or Entity

        """
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
        prop_name : str, optional
            name of a cell property; if None, all cell properties will be returned

        Returns
        -------
        : int or str or dict of {str: any}
            cell property value if `prop_name` is provided, otherwise ``dict`` of all
            cell properties and values
        """
        if prop_name is None:
            return self.edges.get_cell_properties(edge, node)

        return self.edges.get_cell_property(edge, node, prop_name)

    def get_properties(self, id, level=None, prop_name=None):
        """Returns an object's specific property or all properties

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
        if prop_name == None:
            return self.E.get_properties(id, level=level)
        else:
            return self.E.get_property(id, prop_name, level=level)

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

        if edges:
            A, Amap = self.edge_adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self.edge_props.loc[k].to_dict()) for k in Amap]
        else:
            A, Amap = self.adjacency_matrix(s=s, index=True)
            Amaplst = [(k, self.node_props.loc[k].to_dict()) for k in Amap]

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

    def _set_default_state(self):
        """Populate state_dict with default values"""
        self._state_dict = {}

        self._state_dict["dataframe"] = df = self.dataframe
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
        The number of edges of size s that contain node.

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
            return len(self.E.memberships[node])
        else:
            memberships = set()
            for edge in self.E.memberships[node]:
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
        node : hashable or Entity
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
        edge : hashable or Entity
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
        df = self.E.dataframe.pivot(
            index=self.E._data_cols[1],
            columns=self.E._data_cols[0],
            values=self.E._cell_weight_col,
        ).fillna(0)

        if sort_rows:
            df = df.sort_index("index")
        if sort_columns:
            df = df.sort_index("columns")
        if not cell_weights:
            df[df > 0] = 1

        return df

    @classmethod
    @warn_nwhy
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
        if len(M.shape) != (2):
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
