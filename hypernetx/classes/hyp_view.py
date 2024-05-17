# Copyright © 2024 Battelle Memorial Institute
# All rights reserved.
from __future__ import annotations

import warnings
from copy import deepcopy
from collections import UserList
import pandas as pd

warnings.filterwarnings("default", category=DeprecationWarning)

__all__ = ["HypergraphView"]


class HypergraphView(object):
    """
    Wrapper for Property and Incidence Stores holding structural and
    metadata for hypergraph. Provides methods matching EntitySet
    methods in previous versions. Only nodes and edges in the Incidence
    Store will be seeable in this view.
    """

    def __init__(self, incidence_store, level, property_store=None):
        """
        _summary_

        Parameters
        ----------
        incidence_store : IncidenceStore
            All incidence pairs stored as a dataframe
        level : int
            Which type of store: 0 = Edges, 1 = Nodes, 2 = Incidences
        property_store : PropertyStore, optional
            Properties assigned to object in this view, by default None
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

    @property
    def items(self):
        """
        If level 0 or 1, the list of edges or nodes, respectively. If level 2, the IncidenceStore

        Returns
        -------
        IncidenceStore | array
        """
        return set(self._items)

    @property
    def level(self):
        """
        The type of store: 0 = Edges, 1 = Nodes, 2 = Incidences

        Returns
        -------
        int
        """
        return self._level

    @property
    def incidence_store(self):
        """
        IncidenceStore

        Returns
        -------
        IncidenceStore
        """
        return self._incidence_store

    @property
    def property_store(self):
        """
        PropertyStore

        Returns
        -------
        PropertyStore
        """
        return self._property_store

    @property
    def default_weight(self):
        """
        Default weight for an edge, node, or incidence

        Returns
        -------
        int | float
        """
        return self._property_store._default_weight

    @property
    def to_dataframe(self):
        """
        Dataframe of properties (user defined and default) for
        all items in the HypergraphView.
        Creates a properties dataframe of non-user-defined items with default values.
        Combines user-defined and non-user-defined properties into one dataframe.

        Returns
        -------
        pd.DataFrame
        """

        df = self.user_defined_properties.copy(deep=True)

        ### deep copy dictionaries in the misc_properties column
        temp = [deepcopy(d) for d in df.misc_properties.values]
        df.misc_properties = temp

        non_user_defined_items = list(
            set(self._items).difference(self.user_defined_properties.index)
        )

        # skip combining df and non_user_defined_items if non_user_defined_items is empty
        if not non_user_defined_items:
            return df.loc[list(self._items)]

        default_data = self.property_store.default_properties
        default_data.update({"misc_properties": [{}]})
        non_user_defined_properties = pd.DataFrame(
            index=non_user_defined_items, data=default_data
        )

        return pd.concat([df, non_user_defined_properties]).loc[list(self.items)]

    @property  ### will remove this later
    def dataframe(self):
        """
        All properties for objects in the HypergraphView. Same as to_dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return self.to_dataframe

    @property
    def properties(self):
        """
        All properties for objects in the HypergraphView. Same as to_dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return self.to_dataframe

    @property
    def user_defined_properties(self):
        """
        User-defined properties. Does not include items in the HypergraphView
        that the user has not explicitly defined properties for.

        Returns
        -------
        pd.DataFrame
        """
        return self._property_store.properties

    @property
    def memberships(self):
        """
        See :term:`memberships`

        Returns
        -------
        dict
        """
        if self._level == 1 or self._level == 2:
            return self.incidence_store.memberships
        else:
            return {}

    def is_empty(self):
        """
        Returns true if HypergraphView has no edges, nodes, or incidences depending on the level; otherwise, false

        Returns
        -------
        bool
        """
        return len(self._items) == 0

    @property
    def incidence_dict(self):
        """
        incidence dictionary

        Returns
        -------
        dict | None
        """
        if self._level in [0, 2]:
            return self.incidence_store.elements

    @property
    def elements(self):
        """
        See :term:`elements`

        Returns
        -------
        dict
        """
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

        Returns
        -------
        bool
        """
        return item in self._items

    def __call__(self):
        """
        Returns
        -------
        iterator
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
        uid : hashable
            unique identifier for object in HypergraphView

        Returns
        -------
        AttrList
            UserList of incident objects (neighbors in the bipartite graph)
        """
        if uid in self._items:
            neighbors = self.incidence_store.neighbors(self.level, uid)
            return AttrList(uid, self, initlist=neighbors)

    def set_defaults(self, defaults_dict):
        """
        Creates or updates default values in PropertyStore associated with this
        HypergraphView. Does not overwrite existing user-defined properties

        Parameters
        ----------
        defaults_dict : dict
            Dictionary of prop_names to their default values

        Returns
        -------
        None
        """
        self.property_store.set_defaults(defaults_dict)


class AttrList(UserList):
    """Custom list wrapper for integrating PropertyStore data with
    IncidenceStore relationships

    Parameters
    ----------
    hypergraph_view : hypernetx.HypergraphView
    uid : str | int

    Returns
    -------
    AttrList
    """

    def __init__(self, uid, hypergraph_view, initlist=None):
        self._hypergraph_view = hypergraph_view
        self._uid = uid
        if initlist is None:
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
