from typing import Any
from collections.abc import Hashable

from pandas import DataFrame, MultiIndex

ID = "uid"
WEIGHT = "weight"
PROPERTIES = "misc_properties"


class PropertyStore:
    """Class for storing properties of a collection of edges, nodes, or incidences.

    Properties will be stored in a pandas dataframe;

    """
    def __init__(self, data=None, default_weight = 1.0):
        """
        Parameters
        ----------
        data: DataFrame
            optional parameter that holds the properties data in a Dataframe or MultiIndex Dataframe of the following shape
            DataFrame index is uid of objects.
            Example of dataframe (id is set as the index):

                        | weight | misc_properties | <additional property> | ...
            uid
            <edge1uid>   | 1.0 | {} | <property value> | ...

            nodes | weight | misc_properties | <additional property> | ...
             <node1> | 1.0 | {} | <property value> | ...

            Example of multiIndex dataframe (edgeid and nodeid are set as the multiIndex):

                                    | weight | misc_properties | <additional property> | ...
            <edge1uid>  <node1uid>  | 1.0   | {} | <property value> | ...
                        <node2uid>  | 1.5   | {} | <property value> | ...

        """
        # If no dataframe is provided, create an empty dataframe
        if data is None:
            self._data: DataFrame = DataFrame(columns=[ID, WEIGHT, PROPERTIES])
            self._data = self._data.set_index(ID)
        else:
            self._data: DataFrame = data
        self._default_weight = default_weight

        ## minimize space but allow everything to have weight
        # defwt = lambda : {'weight': self._default_weight}
        # self._datadict = defaultdict(defwt)

        # print(self._data)

        # if the index is not set, then set the index on the dataframe
        # if not index:
        #     if isinstance(self._data.index, MultiIndex):
        #         # set index to the first two columns
        #         first_column = self._data.columns[0]
        #         second_column = self._data.columns[1]
        #         self._data.set_index([first_column, second_column], inplace=True)
        #     else:
        #         # set the index to the first column
        #         first_column = self._data.columns[0]
        #         self._data.set_index(first_column, inplace=True)

        # supports the __iter__ magic method
        # self._index_iter = iter(self._data.index)

    @property
    def properties(self) -> DataFrame:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        out: pandas.DataFrame
            a dataframe with the following columns: level, id, uid, weight, properties, <optional props>
        """
        ### update self_data to incorporate the self._datadict
        return self._data

    def get_properties(self, uid) -> dict:
        """Get all properties of an item

        Parameters
        ----------
        uid: Hashable
            uid is the index used to fetch all its properties

        Returns
        -------
        out : dict
            Output dictionary containing all properties of the uid.
            ``{named property: property value, ...,
            properties: {property name: property value}}``

        Raises
        ------
        KeyError
            if (`uid`) is not in :attr:`properties`,

        See Also
        --------
        get_property, set_property
        """
        ### look in self._data and self._datadict
        ### always return properties for a uid in the incidence
        ### store & should should always have a weight property
    
        temp = {'weight': self._default_weight, PROPERTIES : {}}
        if uid in self._data.index:
            properties = self._data.loc[uid].to_dict()
            temp.update(properties)
            temp = flatten(temp)
        return temp

    def get_property(self, uid, prop_name) -> Any:
        """Get a property of an item

        Parameters
        ----------
        uid: Hashable
            uid is the index used to fetch its property

        prop_name : str | int
            name of the property to get

        Returns
        -------
        out : Any
            value of the property

        None
            if property not found

        Raises
        ------
        KeyError
            if (`uid`) is not in :attr:`properties`,

        See Also
        --------
        get_properties, set_property
        """
        properties = self.get_properties(uid)
        return properties.get(prop_name,None)

        # if prop_name in properties:
        #     return properties[prop_name]
        # else:
        #     return properties.get(PROPERTIES,{}).get(prop_name,None)



    def set_property(self, uid, prop_name, prop_val) -> None:
        """Set a property of an item in the 'properties' collection

        Parameters
        ----------
        uid : Hashable
            uid is the index used to set its property
        prop_name : str | int
            name of the property to set
        prop_val : any
            value of the property to set

        Raises
        ------
        KeyError
            If (`uid`) is not in :attr:`properties`

        See Also
        --------
        get_property, get_properties
        """
        ### look in self._data and self._datadict
        properties = self.get_properties(uid)

        if prop_name in properties:
            self._data.loc[uid, prop_name] = prop_val
        # add the new property to the 'properties' column
        else:
            self._data.loc[uid, PROPERTIES].update({prop_name: prop_val})

    # def __iter__(self):
    #     """
    #     Returns an iterator object for iterating over the uid's of the underlying data table

    #     Returns:
    #     -------
    #     out: Iterator
    #         An iterator object for the specified iteration type ('rows' or 'columns').

    #     Example:
    #     --------
    #     >>> iterator = DataFramePropertyStore(dataframe)
    #     >>> for uid in iterator:
    #     ...     print(uid)
    #     """
    #     return self

    # def __next__(self):
    #     try:
    #         return next(self._index_iter)
    #     except StopIteration:
    #         raise StopIteration

    # def __len__(self) -> int:
    #     return len(self._data)

    def __getitem__(self, uid) -> dict: ### get properties with []
        # return self._data.loc[uid].to_dict()
        return self.get_properties(uid)

    def __contains__(self, uid) -> bool: ### do we already have data
        return uid in self._data.index
    

def flatten(my_dict):
    '''Recursive method to flatten dictionary for returning properties as
    a dictionary instead of a Series, from [StackOverflow](https://stackoverflow.com/a/71952620)

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
