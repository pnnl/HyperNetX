# Copyright © 2024 Battelle Memorial Institute
# All rights reserved.
from __future__ import annotations

import warnings

warnings.filterwarnings("default", category=DeprecationWarning)

from copy import deepcopy
from collections import defaultdict, UserList
from collections.abc import Sequence, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix, csr_matrix

from hypernetx.exception import HyperNetXError
from hypernetx.classes.helpers import merge_nested_dicts, dict_depth
from hypernetx.classes.incidence_store import IncidenceStore as IS
from hypernetx.classes.property_store import PropertyStore

__all__ = ["HypergraphView"]


class HypergraphView(object):
    """
    Wrapper for Property and Incidence Stores holding structural and
    meta data for hypergraph. Provides methods matching EntitySet
    methods in previous versions. Only nodes and edges in the Incidence
    Store will be seeable in this view.
    """

    def __init__(self, incidence_store, level, property_store=None):
        """
        _summary_

        Parameters
        ----------
        incidence_store : _type_
            _description_
        level : _type_
            _description_
        property_store : _type_, optional
            _description_, by default None
        """
        self._incidence_store = incidence_store
        self._level = level
        self._property_store = property_store

        ### incidence store needs index or columns
        if level == 0:
            self._items = self._incidence_store.edges
        elif level == 1:
            self._items = self._incidence_store.nodes
        elif level == 2:
            self._items = self._incidence_store

        # self._properties = PropertyStore()
        ### if no properties and level 0 or 1,
        ### create property store that
        ### returns weight 1 on every call for a weight
        ### and empty properties otherwise.

    @property
    def level(self):
        return self._level

    @property
    def incidence_store(self):
        return self._incidence_store

    @property
    def property_store(self):
        return self._property_store

    @property
    def to_dataframe(self):
        """
        Dataframe of properties (user defined and default) for 
        all items in the HypergraphView.

        Returns
        -------
        out: pd.DataFrame
        """
        # Create a properties dataframe of non-user-defined items with default values
        df = self.properties.copy(deep=True)
        ### deep copy dictionaries in the misc_properties column
        temp = [deepcopy(d) for d in df.misc_properties.values]
        df.misc_properties = temp

        non_user_defined_items = list(
            set(self._items).difference(self.properties.index)
        )
        default_data = self.property_store.default_properties_df()
        non_user_defined_properties = pd.DataFrame(
            index=non_user_defined_items, data=default_data
        )

        # Combine user-defined and non-user-defined properties into one dataframe
        return pd.concat([df, non_user_defined_properties])

    @property  ### will remove this later
    def dataframe(self):
        return self._property_store.properties

    @property
    def properties(self):
        """
        User-defined properties. Does not include items in the HypergraphView
        that the user has not explicitly defined properties for.

        Returns
        -------
        out: pd.DataFrame
        """
        return self._property_store.properties

    @property
    def memberships(self):
        if self._level == 1 or self._level == 2:
            return self.incidence_store.memberships
        else:
            return {}

    def is_empty(self):
        return len(self._items) == 0

    @property
    def incidence_dict(self):
        if self._level in [0, 2]:
            return self.incidence_store.elements

    @property
    def elements(self):
        if self._level == 0 or self._level == 2:
            return self.incidence_store.elements
        else:
            return {}

    def __iter__(self):
        """
        Defined by level store
        iterate over the associated level in the incidence store
        """
        return iter(self._items)

    def __len__(self):
        """
        Defined by incidence store
        """
        return len(self._items)

    def __contains__(self, item):
        """
        Defined by incidence store

        Parameters
        ----------
        item : _type_
            _description_
        """
        return item in self._items

    def __call__(self):
        """
        Returns iterator
        """
        return iter(self._items)

    def __getitem__(self, uid):
        """
        Returns incident objects (neighbors in bipartite graph)
        to keyed object as an AttrList.
        Returns AttrList associated with item,
        attributes/properties may be called
        from AttrList
        If level 0 - elements, if level 1 - memberships,
        if level 2 - TBD - uses getitem from stores and links to props
            These calls will go to the neighbors method in the incidence store

        Parameters
        ----------
        uid : _type_
            _description_
        """
        if uid in self._items:
            neighbors = self.incidence_store.neighbors(self.level, uid)
            return AttrList(uid, self, initlist=neighbors)

    # def properties(self,key=None,prop_name=None):
    #     """
    #     Return dictionary of properties or single property for key
    #     Currently ties into AttrList object in utils.
    #     Uses getitem from property stores

    #     Parameters
    #     ----------
    #     key : _type_
    #         _description_
    #     prop_name : _type_, optional
    #         _description_, by default None

    #     Returns
    #     -------
    #     if key=None and prop=None, dictionary key:properties OR
    #     elif prop=None, dictionary prop: value for key
    #     elif key = None, dictionary of keys: prop value
    #     else property value
    #     """
    #     # If a dfp style dataframe use .to_dict()
    #     pass

    # def __call__(self):
    #     """
    #     Iterator over keys in store -
    #     level 0 = edges, 1 = nodes, 2 = incidence pairs
    #     Uses incidence store.
    #     """
    #     pass

    # def __getattribute__(self, __name: str) -> Any:
    #     pass

    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     pass

    # def to_json(self):
    #     """
    #     Returns jsonified data. For levels 0,1 this will be the edge and nodes
    #     properties and for level 2 this will be the incidence pairs and their
    #     properties
    #     """
    #     pass

    # #### data,labels should be handled in the stores and accessible
    # #### here - if we want them??
    # def encoder(self,item=None):
    #     """
    #     returns integer encoded data and labels for use with fast
    #     processing methods in form of label:int dictionaries
    #     """
    #     pass

    # def decoder(self):
    #     """
    #     returns int:label dictionaries
    #     """


class AttrList(UserList):
    """Custom list wrapper for integrating PropertyStore data with
    IncidenceStore relationships

    Parameters
    ----------
    hypergraph_view : hypernetx.HypergraphView
    uid : str | int

    Returns
    -------
        : AttrList object
    """

    def __init__(self, uid, hypergraph_view, initlist=None):
        self._hypergraph_view = hypergraph_view
        self._uid = uid
        if initlist == None:
            initlist = hypergraph_view._incidence_store.neighbors(self._level, uid)
        super().__init__(initlist)

    def __getattr__(self, attr: str):
        """Get attribute value from properties of :attr:`entity`

        Parameters
        ----------
        attr : str

        Returns
        -------
        any
            attribute value; None if not found
        """
        if attr == "memberships":
            if self._level == 1:
                return self.data
            else:
                return []
        elif attr == "elements":
            if self._level == 0:
                return self.data
            else:
                return []
        elif attr == "properties":
            return self._hypergraph_view._property_store.get_properties(self._uid)
        else:
            return self._hypergraph_view._property_store.get_property(self._uid, attr)

    def __setattr__(self, attr, val):
        """Set attribute value in properties

        Parameters
        ----------
        attr : str
        val : any
        """
        if attr in ["_hypergraph_view", "_uid", "data"]:
            object.__setattr__(self, attr, val)
        else:
            self._hypergraph_view._property_store.set_property(self._uid, attr, val)


def flatten(my_dict):
    """Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)
    Redundant keys are kept in the order of hierarchy (first seen kept by default)

    """
    result = {}
    for key, value in my_dict.items():
        if isinstance(value, dict):
            temp = flatten(value)
            temp.update(result)
            result = temp
        else:
            result[key] = value
    return result
