from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

UID = "uid"
WEIGHT = "weight"
PROPERTIES = "properties"


class PropertyStore(ABC):
    @property
    @abstractmethod
    def properties(self) -> Any:
        """Properties assigned to items in the underlying data table"""
        ...

    @abstractmethod
    def get_properties(self, uid) -> dict[Any, Any]:
        """Get all properties of an item

        Parameters
        ----------
        uid : tuple[str | int] | tuple[str | int, str | int ]
            edge name, node name, or edge-node pair

        Returns
        -------
        prop_vals : dict
            ``{named property: property value, ...,
            misc. property column name: {property name: property value}}``

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
    def get_property(self, uid, prop_name) -> Any:
        """Get a property of an item

        Parameters
        ----------
        uid : tuple[str | int] | tuple[str | int, str | int ]
            edge name, node name, or edge-node pair
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
    def set_property(self, uid, prop_name, prop_val) -> None:
        """Set a property of an item in the 'properties' dictionary

        Parameters
        ----------
        uid : tuple[str | int] | tuple[str | int, str | int ]
            edge name, node name, or edge-node pair
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

    def __iter__(self) -> iter:
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
        uid : tuple[str | int] | tuple[str | int, str | int ]
            edge name, node name, or edge-node pair

        Returns
        -------
        dict
        """
        ...

    def __contains__(self, uid) -> bool:
        """Returns true if uid is in the underlying data table; false otherwise

        Parameters
        ----------
        uid : tuple[str | int] | tuple[str | int, str | int ]
            edge name, node name, or edge-node pair

        Returns
        -------
        bool
        """
        ...


class DataFramePropertyStore(PropertyStore):
    """
    data will be a dataframe of the following shape

    uid | weight | properties | ...
    (edge1) | 1.0 | {} | somepropname | somepropname2| ...

    uid | weight | properties | ...
    (node1) | 1.0 | {} | somepropname | somepropname2| ...

    uid | weight | properties | ...
    (edge1, node1) | 1.0 | {} | somepropname | somepropname2| ...


     uid | weight | properties | ...
    (edge1, node1) | 1.0 | {} | somepropname | somepropname2| ...
    (edge1, node2) | 1.0 | {} | somepropname | somepropname2| ...

    """

    def __init__(self, data: pd.DataFrame):
        self._data: pd.DataFrame = data

    @property
    def properties(self) -> pd.DataFrame:
        """Properties assigned to items in the underlying data table

        Returns
        -------
        pandas.DataFrame a dataframe with the following columns: uid, weight, properties, ...
        """
        return self._data

    def get_properties(self, uid) -> dict[Any, Any]:
        row_idx = self._uid_index(uid)
        props = self._data.loc[row_idx].to_dict()
        props.pop(UID)
        return props

    def get_property(self, uid, prop_name) -> Any:
        row_idx = self._uid_index(uid)
        props = self._data.loc[row_idx].to_dict()

        if prop_name in props.get(PROPERTIES):
            return props.get(PROPERTIES).get(prop_name)
        if prop_name not in props:
            return None
        return props.pop(prop_name)

    def set_property(self, uid, prop_name, prop_val) -> None:
        row_idx = self._uid_index(uid)

        if prop_name in self._data.columns and prop_name != PROPERTIES:
            self._data.loc[row_idx, prop_name] = prop_val
        else:
            self._data.loc[row_idx, PROPERTIES].update({prop_name: prop_val})

    def _uid_index(self, uid) -> int:
        if uid not in self:
            raise KeyError(f"uid, ({','.join(uid)}), not found in PropertyStore")
        row_idx = self._data.index[self._data[UID] == uid].tolist()
        return row_idx[0]

    def __iter__(self) -> iter:
        return self._data.itertuples(name="PropertyStore", index=False)

    def __len__(self) -> int:
        return len(self._data.index)

    def __getitem__(self, uid) -> dict:
        row_idx = self._uid_index(uid)
        props = self._data.loc[row_idx].to_dict()
        props.pop(UID)
        return props

    def __contains__(self, uid) -> bool:
        idx = self._data.index[self._data[UID] == uid]
        return not idx.empty


# TODO: Implement the dictionary version of PropertyStore
# class DictPropertyStore(PropertyStore):
#     @property
#     def properties(self) -> dict:
#         pass
#
#     def get_properties(self, uid) -> dict[Any, Any]:
#         pass
#
#     def get_property(self, uid, prop_name) -> Any:
#         pass
#
#     def set_property(self, uid, prop_name, prop_val) -> None:
#         pass
