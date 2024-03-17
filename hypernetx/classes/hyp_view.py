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
from hypernetx.classes.property_store import PropertyStore

__all__ = ["HypergraphView"]

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
        self._incidence_store = incidence_store
        self._level = level
        self._property_store = property_store


        ### incidence store needs index or columns
        if level == 0 :
            self._items = self._incidence_store.edges
        elif level == 1 :
            self._items = self._incidence_store.nodes
        elif level == 2 :
            self._items = self._incidence_store.data.values

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
    def dataframe(self):
        return self._property_store._data

    @property
    def properties(self):
        return self._property_store.properties

    @property
    def memberships(self):
        if self._level == 1 or self._level == 2:
            return self.incidence_store.memberships
        else:
            return {}
        
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

    def __contains__(self,item):
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
        uid : _type_
            _description_
        """
        if uid in self._items:
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
    hypergraph_view : hypernetx.HypergraphView
    uid : str | int

    Returns
    -------
        : NList object 
    """

    def __init__(
        self,
        hypergraph_view,
        uid,
        initlist = None
    ):
        self.__dict__['_hypergraph_view'] = hypergraph_view
        self.__dict__['_props'] = hypergraph_view._property_store.get_properties(uid)   
        self._level = hypergraph_view._level
        self._uid = uid
        if initlist == None:
            initlist = hypergraph_view._incidence_store.neighbors(self._level,uid)
        super().__init__(initlist)
    
    def __getattr__(self, attr: str = None):
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
            return self._props
        else:
            return self._props.get(attr,None)



    # def __setattr__(self, attr: str , val: Any = None) -> None:
    #     """Set attribute value in properties 

    #     Parameters
    #     ----------
    #     attr : str
    #     val : any
    #     """
    # #     temp = self._hypergraph_view
    # #     temp.property_store.set_property(self.uid, attr, val)
    # #     self.__dict__['_hypergraph_view'] = temp
    # #     self.__dict__['_props'] = temp._property_store.get_properties(self._uid)
    #     if attr in ["_level", "_uid"]:
    #         pass
    #         # object.__setattr__(self, attr, val)
    #     else:
    #         self._hypergraph_view._property_store.set_property(self._uid, attr, val)

    # # def properties(self):
    # #     """
    # #     Return dict of properties associated with this AttrList as a dictionary.
    # #     """
    #     # pass


def flatten(my_dict):
    '''Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)
    Redundant keys are kept in the order of hierarchy (first seen kept by default)

    '''
    result = {}
    for key, value in my_dict.items():
        if isinstance(value, dict):
            temp = flatten(value)
            temp.update(result)
            result = temp
        else:
            result[key] = value
    return result
