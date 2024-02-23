from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Hashable

import pandas as pd

UUID = "uuid"
WEIGHT = "weight"
PROPERTIES = "properties"


class PropertyStore(ABC):
    @property
    @abstractmethod
    def properties(self) -> Any:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        Object containing all properties of the underlying data table
        """
        ...

    @abstractmethod
    def get_properties(self, uid: Hashable) -> dict[Any, Any]:
        """Get all properties of an item

        Parameters
        ----------
        uid : Hashable
            uid is the index used to fetch all its properties

        Returns
        -------
        prop_vals : dict
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
        ...

    @abstractmethod
    def get_property(self, uid: Hashable, prop_name: str | int) -> Any:
        """Get a property of an item

        Parameters
        ----------
        uid : Hashable
            uid is the index used to fetch its property
        prop_name : str | int
            name of the property to get

        Returns
        -------
        prop_val : any
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
        ...

    @abstractmethod
    def set_property(self, uid: Hashable, prop_name: str | int, prop_val: Any) -> None:
        """Set a property of an item in the 'properties' colelction

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

    def __iter__(self):
        """
        Returns an iterator object for iterating over data in PropertyStore

        Returns:
        -------
        DataIterator:
            An iterator object
        """
        return self

    def __iter__(self):
        """Returns an iterator over items in the underlying data table

        Returns
        -------
        Iterator

        """
        ...

    def __len__(self) -> int:
        """Number of items in the underlying data table

        Returns
        -------
        int
        """
        ...

    def __getitem__(self, uid) -> dict:
        """Returns the common attributes (e.g. weight) and properties of an item in the underlying data table

        Returns same result as get_property

        Parameters
        ----------
        Hashable
            uid is the index used to set its property

        Returns
        -------
        dict
        """
        ...

    def __contains__(self, uid) -> bool:
        """Returns true if uid is in the underlying data table; false otherwise

        Parameters
        ----------
        Hashable
            uid is the index used to set its property

        Returns
        -------
        bool
        """
        ...


class DataFramePropertyStore(PropertyStore):
    def __init__(self, data: pd.DataFrame):
        """
        Parameters
        ----------
        data: pd.DataFrame
            data must be a MultiIndex Dataframe of the following shape

            level | id | weight | properties | ...
            <common level value> | <edge> | 1.0 | {} | somepropname | somepropname2| ...

            level | id | weight | properties | ...
            <common level value> | <node1> | 1.0 | {} | somepropname | somepropname2| ...

            level | id | weight | properties | ...
            <edge> | <node> | 1.0 | {} | somepropname | somepropname2| ...
        """
        self._data: pd.DataFrame = data
        self._index_iter = iter(self._data.index)

    @property
    def properties(self) -> pd.DataFrame:
        """Properties assigned to all items in the underlying data table

        Returns
        -------
        pandas.DataFrame a dataframe with the following columns: level, id, uid, weight, properties, <optional props>
        """
        return self._data

    def get_properties(self, uid: tuple[str | int, str | int]) -> dict[Any, Any]:
        """Get all properties of an item

        Parameters
        ----------
        uid: tuple[str | int, str | int ]
            uid is the index used to fetch all its properties

        Returns
        -------
        prop_vals : dict
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
            raise KeyError(f"uid, ({','.join(uid)}), not found in PropertyStore")
        return properties.to_dict()

    def get_property(self, uid: tuple[str | int, str | int], prop_name: str | int) -> Any:
        """Get a property of an item

        Parameters
        ----------
        uid: tuple[str | int, str | int ]
            uid is the index used to fetch its property

        prop_name : str | int
            name of the property to get

        Returns
        -------
        prop_val : any
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
        uid : tuple[str | int] | tuple[str | int, str | int ]
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
        Iterator:
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

    def __getitem__(self, uid: tuple[str | int, str | int]) -> dict:
        return self._data.loc[uid].to_dict()

    def __contains__(self, uid: tuple[str | int, str | int]) -> bool:
        return uid in self._data.index


# TODO: Implement DictPropertyStore(PropertyStore), which uses a dictionary to store properties
