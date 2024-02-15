# Copyright © 2024 Battelle Memorial Institute
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
from hypernetx.classes.incidence_store import IncidenceStore as IS


__all__ = ["Hypergraph"]
T = TypeVar("T", bound=Union[str, int])


##################### PROXY CLASSES FOR CONSTRUCTION
class PropertyStore(object):
    def __init__(self,dfp=None):
        if dfp is not None:
            self._dataframe = dfp
            self._dataframe['properties'] = [{} for idx in self.dataframe.index]
        else:
            self._dataframe = pd.DataFrame(columns=['id','weight','properties'])
    def __call_(self):
        return self

    @property
    def dataframe(self):
        return self._dataframe

class IncidenceStore():
    def __init__(self,df):
        self._dataframe = df
        self.proxy = IS(df)
            
    def __call__(self):
        return self
    
    def __iter__(self):
        return self.proxy.__iter__()
    
    @property
    def edges(self):
        return self.proxy.edges
    
    @property
    def nodes(self):
        return self.proxy.nodes

    @property
    def dataframe(self):
        return self._dataframe
    
######################################################

class HypergraphView(object):
    """
    Wrapper for Property and Incidence Stores holding structural and 
    meta data for hypergraph. Provides methods matching EntitySet 
    methods in previous versions.
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
        self.name = "test"
        self._incidence_store = incidence_store
        ### incidence store needs index or columns
        self._level = level  ## edges, nodes, or incidence pairs
        self._properties = property_store or PropertyStore()

        # self._properties = PropertyStore()  
        ### if no properties and level 0 or 1, 
        ### create property store that 
        ### returns weight 1 on every call for a weight 
        ### and empty properties otherwise.

    @property
    def dataframe(self):
        return self._incidence_store.dataframe
    
    def __iter__(self):
        """
        Defined by level store
        iterate over the associated level in the incidence store
        """
        level = self._level
        if level == 0:
            return self._incidence_store.edges.__iter__()
        elif level == 1:
            return self._incidence_store.nodes.__iter__()
        elif level == 2:
            return self._incidence_store.__iter__()

    def __len__(self):
        """
        Defined by incidence store 
        """
        level = self._level
        if 

    # def __contains__(self,item):
    #     """
    #     Defined by level store

    #     Parameters
    #     ----------
    #     item : _type_
    #         _description_
    #     """
    #     pass

    
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



    # def __getitem__(self,key):
    #     """
    #     Returns incident objects (neighbors in bipartite graph) 
    #     to keyed object as an AttrList.
    #     Returns AttrList associated with item, 
    #     attributes/properties may be called 
    #     from AttrList 
    #     If level 0 - elements, if level 1 - memberships,
    #     if level 2 - TBD - uses getitem from stores and links to props
    #         These calls will go to the neighbors method in the incidence store

    #     Parameters
    #     ----------
    #     key : _type_
    #         _description_
    #     """
    #     pass

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