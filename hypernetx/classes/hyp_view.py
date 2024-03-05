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

from hypernetx.classes import EntitySet
from hypernetx.exception import HyperNetXError
from hypernetx.utils.decorators import warn_nwhy
from hypernetx.classes.helpers import merge_nested_dicts, dict_depth
from hypernetx.classes.incidence_store import IncidenceStore as IS


__all__ = ["HypergraphView"]

##################### PROXY CLASSES FOR CONSTRUCTION
# class PropertyStore(object):
#     """
#     Wrapper for a pandas dataframe. Minimal logic but limits changes
#     to

#     Parameters
#     ----------
#     object : _type_
#         _description_
#     """
#     def __init__(self,dfp=None, level=0):
#         if dfp is not None:
#             self._dataframe = dfp
#             if 'properties' not in dfp.columns:
#                 self._dataframe['properties'] = [{} for idx in self.dataframe.index]
#             else:
#                 dfp.properties.fillna({})
#             if 'weight' not in dfp.columns:
#                 self._dataframe.weight = 1
#             else:
#                 dfp.weight.fillna(1)
#             if level in [0,1]:
#                 self._dataframe = self._dataframe.set_index(self._dataframe.columns[0])
#             elif level in [2]:
#                 self._dataframe = self._dataframe.set_index([self._dataframe.columns[0],self._dataframe.columns[1]])
#         else:
#             self._dataframe = pd.DataFrame(columns=['weight','properties'])


#     def __call__(self):
#         return self

#     def __iter__(self):
#         return iter(self._dataframe.index)

#     def __len__(self):
#         return len(self._dataframe)

#     @property
#     def dataframe(self):
#         return self._dataframe

#     def get_property(self, uid, prop_name):
#         prop_val = None
#         df = self.dataframe
#         try:
#             prop_val = df.loc[uid][prop_name]
#         except KeyError:
#             prop_val = df.loc[uid]['properties'].get(prop_name,None)
#         return prop_val

# class IncidenceStore(IS):
#     def __init__(self,df):
#         super().__init__(df)
#         # self._dataframe = self._data
#         # self.proxy = IS(df)

#     # def __call__(self):
#     #     return self

#     # def __iter__(self):
#     #     return self.proxy.__iter__()

#     # @property
#     # def edges(self):
#     #     return self.proxy.edges

#     # @property
#     # def nodes(self):
#     #     return self.proxy.nodes

#     @property
#     def dataframe(self):
#         return self._data

######################################################

class HypergraphView(object):
    """
    Wrapper for Property and Incidence Stores holding structural and
    meta data for hypergraph. Provides methods matching EntitySet
    methods in previous versions. Only nodes and edges in the Incidence
    Store will be seeable in this view.
    """
    def __init__(self,incidence_store,level,property_store=None):
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
        self._store = incidence_store
        self._level = level
        if property_store is not None:
            self._props = property_store

        else:
            self._props = PropertyStore()


        ### incidence store needs index or columns
        if level == 0 :
            self._items = self._store.edges
        elif level == 1 :
            self._items = self._store.nodes
        elif level == 2 :
            self._items = self._store.data.values

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
        return self._store

    @property
    def property_store(self):
        return self._props

    @property
    def properties(self):
        return self._props.properties


    # @property
    # def levelset(self):
    #     """
    #     _summary_

    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """
    #     level = self.level
    #     if level == 0:
    #         return self._store.edges.unique()
    #     elif level == 1:
    #         return self._store.nodes.unique()
    #     elif level == 2:
    #         return self._store.dataframe


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
        return len(self._itemse)

    def __contains__(self,item):
        """
        Defined by incidence store

        Parameters
        ----------
        item : _type_
            _description_
        """
        return item in self._items


    # def to_dataframe(self):
    #     """
    #     Defined by property store.
    #     Returns a pandas dataframe keyed by level keys
    #         with properties as columns or in a variable length dict.
    #         The returned data frame will either reflect the
    #         """
    #     pass



    # def to_dict(self, data=False):
    #     """
    #     Association dictionary - neighbors from bipartite form
    #     returns a dictionary of key: <elements,memberships,elements>
    #     for level 0,1,2
    #     values are initlist from AttrList class
    #     if data = True, include data
    #     """
    #     pass



    def __getitem__(self,uid):
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
        key : _type_
            _description_
        """
        level = self._level
        if level == 0 or level == 1:
            elements = self._store.neighbors(level,uid)
        else:
            elements = None
        return NList(self, uid)


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

    # def memberships(self,item):
    #     """
    #     applies to level 1: returns edges the item belongs to.
    #     if level = 0 or 2 then memberships returns none.

    #     Parameters
    #     ----------
    #     item : _type_
    #         _description_
    #     """
    #     pass

    # def elements(self,item):
    #     """
    #     applies to levels 0: returns nodes the item belongs to.
    #     if level = 1 or 2 then elements returns none.

    #     Parameters
    #     ----------
    #     item : _type_
    #         _description_
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



class NList(UserList):
    """Custom list wrapper for integrating PropertyStore data with
    IncidenceStore relationships

    Parameters
    ----------
    entity : hypernetx.EntitySet
    key : tuple of (int, str or int)
        ``(level, item)``
    initlist : list, optional
        list of elements, passed to ``UserList`` constructor

    # New Parameters
    # --------------
    # key :
    # property_store :
    # incidence_store :

    # methods return curren view of properties and
    # neighbors
    """

    def __init__(
        self,
        hypergraph_view,
        uid,
    ):
        self._props = hypergraph_view._props
        self._level = hypergraph_view._level
        self._uid = uid
        initlist = hypergraph_view._store.neighbors(self._level,uid)
        super().__init__(initlist)



    def __getattr__(self, attr: str) -> Any:
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
            return self._props.get_properties(self._uid)
        else:
            return self._props.get_property(self._uid, attr)



    # def __setattr__(self, attr: str, val: Any) -> None:
    #     """Set attribute value in properties of :attr:`entity`

    #     Parameters
    #     ----------
    #     attr : str
    #     val : any
    #     """
    #     if attr in ["_entity", "_key", "data"]:
    #         object.__setattr__(self, attr, val)
    #     else:
    #         self._entity.set_property(self._key[1], attr, val, level=self._key[0])

    # def properties(self):
    #     """
    #     Return dict of properties associated with this AttrList as a dictionary.
    #     """
    #     pass


def flatten(my_dict):
    '''Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)

    '''
    result = {}
    for key, value in my_dict.items():
        if isinstance(value, dict):
            result.update(_flatten(value))
        else:
            result[key] = value
    return result
