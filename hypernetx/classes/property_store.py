import uuid

from typing import Any
from collections.abc import Hashable

from pandas import DataFrame

UID = "uid"
ID = "id"
WEIGHT = "weight"
PROPERTIES = "misc_properties"


class PropertyStore:
    """Class for storing properties of a collection of edges, nodes, or incidences.

    Properties will be stored in a pandas dataframe;

    """
    def __init__(self, data=None, index=False):
        """
        Parameters
        ----------
        data: DataFrame
            optional parameter that holds the properties data in a Dataframe or MultiIndex Dataframe of the following shape

            Example of dataframe (id is set as the index):

            edges | weight | misc_properties | <additional property> | ...
             <node1> | 1.0 | {} | <property value> | ...

            nodes | weight | misc_properties | <additional property> | ...
             <node1> | 1.0 | {} | <property value> | ...

            Example of multiIndex dataframe (level and id are set as the multiIndex):

            edges | nodes | weight | misc_properties | <additional property> | ...
            <edge> | <node> | 1.0 | {} | <property value> | ...

        index: Boolean
            optional parameter to use the dataframe index as the uid
            defaults to False
        """
        # If no dataframe is provided, create an empty dataframe
        if data is None:
            self._data: DataFrame = DataFrame(columns=[ID, WEIGHT, PROPERTIES])
            self._data.set_index(ID)
        else:
            self._data: DataFrame = data

        # add the UID column to the dataframe based on 'index'
        if index:
            self._data[UID] = self._data.index
        else:
            self._data[UID] = [str(uuid.uuid4()) for _ in range(len(self._data))]

        # supports the __iter__ magic method
        self._index_iter = iter(self._data.index)

    @property
    def properties(self) -> DataFrame:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        out: pandas.DataFrame
            a dataframe with the following columns: level, id, uid, weight, properties, <optional props>
        """
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
        try:
            properties = self._data.loc[uid]
        except KeyError:
            raise KeyError(f"uid, {uid}, not found in PropertyStore")
        return properties.to_dict()

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

        if prop_name in self._data.columns:
            return properties.get(prop_name)

        props_collection = properties.get(PROPERTIES)
        if prop_name in props_collection:
            return props_collection.get(prop_name)

        return None

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
        properties = self.get_properties(uid)

        if prop_name in properties:
            self._data.loc[uid, prop_name] = prop_val
        # add the new property to the 'properties' column
        else:
            self._data.loc[uid, PROPERTIES].update({prop_name: prop_val})

    def __iter__(self):
        """
        Returns an iterator object for iterating over the uid's of the underlying data table

        Returns:
        -------
        out: Iterator
            An iterator object for the specified iteration type ('rows' or 'columns').

        Example:
        --------
        >>> iterator = DataFramePropertyStore(dataframe)
        >>> for uid in iterator:
        ...     print(uid)
        """
        return self

    def __next__(self):
        try:
            return next(self._index_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, uid) -> dict:
        return self._data.loc[uid].to_dict()

    def __contains__(self, uid) -> bool:
        return uid in self._data.index
