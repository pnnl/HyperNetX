# Copyright © 2024 Battelle Memorial Institute
# All rights reserved.
from __future__ import annotations

import pandas as pd

__all__=["IncidenceStore"]

class IncidenceStore:
    """
    Incidence store object that stores and accesses (multi) incidences with standard methods.

    Parameters
    ----------
    data : Two column pandas dataframe of edges and nodes, respectively.
    """

    def __init__(self, data):
        """
        Initiate data in self as the two column pandas dataframe provided through factory method.

        Parameters
        ----------
        data : _type_
            collection of ordered pairs
        """
        # initiate self with data (pandas dataframe) with duplicate incidence pairs removed.
        self._data = data
        self._elements = data.groupby('edges').agg(list).to_dict()['nodes']
        self._memberships = data.groupby('nodes').agg(list).to_dict()['edges']


    @property
    def data(self):
        return self._data
    
    @property
    def elements(self):
        return self._elements
    
    @property
    def memberships(self):
        return self._memberships

    @property
    def dimensions(self):
        """
        Dimensions of incidence pairs dataframe.
        i.e., the number of distinct nodes and edges.

        Returns
        -------
        tuple of ints
             Tuple of size two of (number of unique nodes, number of unique edges).
        """
        return (len(self._elements), len(self._memberships))

    @property
    def edges(self):
        """
        Returns an array of edge names from the incidence pairs

        Returns
        -------
        array
             Returns an array of edge names
        """
        return list(self._data['edges'].unique())

    @property
    def nodes(self):
        """
        Returns an array of node names from the incidence pairs

        Returns
        -------
        array
             Returns an array of node names
        """
        return list(self._data['nodes'].unique())

    def __iter__(self):
        """
        Iterator over the incidence pairs of the hypergraph

        Returns
        -------
        iter of tuples
            Iterator over incidence pairs (tuples) in the hypergraph.
        """
        # itertuples provides iterator over rows in a dataframe 
        # with index as false to not return index
        # and name as None to return a standard tuple.
        return iter(self._data.itertuples(index=False, name=None))

    def __len__(self):
        """
        Total number of incidences

        Returns
        -------
        int
            Number of incidence pairs in the hypergraph.
        """

        return len(self._data)

    def __contains__(self, incidence_pair):
        """
        Checks if an incidence pair exists in the incidence pairs dataframe.
        First, this checks if the incidence pair is of length two.
        Then, it checks if it exists in the incidence pairs.

        Parameters
        ----------
        incidence_pair : tuple
           Incidence pair that is a tuple (or array-like object; e.g., list or array) of length two.

        Returns
        -------
        bool
            True if incidence pair exists in incidence store.
        """
        # df = self._data

        # #verify the incidence pair is of length two. Otherwise, pair does not exist.
        # if len(incidence_pair) == 2:
        #     node, edge = incidence_pair[0], incidence_pair[1]
        #     # check if first element in pair (node) exists in 'nodes' column anywhere
        #     # and check if second element of pair (edge) exists in 'edges' column anywhere.
        #     does_contain = ((df['nodes'] == node) & (df['edges'] == edge)).any()
        #     return does_contain
        # else:
        #     return False
        
        return incidence_pair in self._data.values



    def neighbors(self, level, key):
        """
        Returns elements or memberships depending on level.

        Parameters
        ----------
        level : int
            Level indicator for finding either elements or memberships.
            For level 0 (elements), returns nodes in the edge.
            For level 1 (memberships), returns edges containing the node.
        key : int or str
            Name of node or edge depending on level.

        Returns
        -------
        list
            Elements or memberships (depending on level) of a given edge or node, respectively.
        """
        # df = self._data

        # if level == 0: # if looking for elements
        #     try:
        #         # Group by 'edges' and get 'nodes' within each group where 'edges' matches the key
        #         return df.groupby('edges')['nodes'].get_group(key).tolist()
        #     except KeyError:
        #         # Return empty list if key doesn't exist for level 0 (edge)
        #         return []
        # elif level == 1: # if looking for memberships
        #     try:
        #         # Group by 'nodes' and get 'edges' within each group where 'nodes' matches the key
        #         return df.groupby('nodes')['edges'].get_group(key).tolist()
        #     except KeyError:
        #         # Return empty list if key doesn't exist for level 1 (node)
        #         return []
        # elif level == 2:
        #     return []
        # else:
        #     return []
        
        if level == 0:
            return self._elements.get(key,[])
        elif level == 1:
            return self.__memberships.get(key,[])
        else:
            return []
        

    def restrict_to(self, level, items, inplace=False):
        ### TODO if inplace == True the constructor's attributes need to be
        ### adjusted.

        """
        returns IncidenceStore of subset of incidence store restricted
        to pairs with items in the given level
        Will return with same data or deepcopy depending on inplace

        Parameters
        ----------
        level : int
            Level indicator for finding either elements or memberships.
            For level 0 (elements), returns nodes in the edge.
            For level 1 (memberships), returns edges containing the node.
        items : list
            List of uids to be removed from level
        inplace : bool, optional
            whether to replace self, by default False

        Returns
        -------
        list
            subset of incidence store given a restriction.
        """


        if level == 0:
            column = 'edges'
        elif level == 1:
            column = 'nodes'
        else:
            raise ValueError("Invalid level provided. Must be 0 or 1.")

        if inplace:
            self._data.drop(self._data[~self._data[column].isin(items)].index, inplace=True)
            return self._data

        else: #return a subset without editing the original dataframe.
            df = self._data
            return df[df[column].isin(items)]
