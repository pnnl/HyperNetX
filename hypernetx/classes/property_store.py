from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Any

import pandas as pd


@dataclass
class PropertyStore(ABC):
    data: Any = None

    @property
    @abstractmethod
    def properties(self) -> Any:
        """Properties assigned to items in the underlying data table"""
        ...

    @abstractmethod
    def get_properties(self, key) -> dict[Any, Any]:
        """Get all properties of an item

        Parameters
        ----------
        key : str | int

        Returns
        -------
        prop_vals : dict
            ``{named property: property value, ...,
            misc. property column name: {property name: property value}}``

        Raises
        ------
        KeyError
            if (`key`) is not in :attr:`properties`,

        See Also
        --------
        get_property, set_property
        """
        ...

    @abstractmethod
    def get_property(self, key, prop_name) -> Any:
        """Get a property of an item

        Parameters
        ----------
        key : str | int
            name of an item
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
            if (`key`) is not in :attr:`properties`,

        See Also
        --------
        get_properties, set_property
        """
        ...

    @abstractmethod
    def set_property(self, key, prop_name, prop_val) -> None:
        """Set a property of an item

        Parameters
        ----------
        key : str | int
            name of an item
        prop_name : str | int
            name of the property to set
        prop_val : any
            value of the property to set

        Raises
        ------
        ValueError
            If `key` is not in :attr:`properties`

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

    def __getitem__(self, key: Union[str, int]) -> dict:
        """Returns the common attributes (e.g. weight) and properties of a key in the underlying data table

        Parameters
        ----------
        key : str | int

        Returns
        -------
        dict
        """
        ...

    def __contains__(self, key: Union[str, int]) -> bool:
        """Returns true if key is in the underlying data table; false otherwise

        Parameters
        ----------
        key : str | int

        Returns
        -------
        bool
        """
        ...

    def __getattr__(self, key: Union[str, int]) -> dict:
        """Returns the properties of a key in the underlying data table

        Parameters
        ----------
        key : str | int

        Returns
        -------
        dict

        """
        ...

    def __setattr__(self, key: Union[str, int], value: dict) -> None:
        """Sets the properties of a key in the underlying data table

        Parameters
        ----------
        key : str | int
        value : dict

        Returns
        -------
        None
        """
        ...


# TODO: Implement abstract methods and dunder methods using EntitySet code snippets
class DataFramePropertyStore(PropertyStore):
    @property
    def properties(self) -> pd.DataFrame:
        """Properties assigned to items in the underlying data table

        Returns
        -------
        pandas.DataFrame a dataframe with the following columns: level/(edge|node), uid, weight, properties
        """
        return self.data

    def get_properties(self, key) -> dict[Any, Any]:
        pass

    def get_property(self, key, prop_name) -> Any:
        pass

    def set_property(self, key, prop_name, prop_val) -> None:
        pass

    def __iter__(self) -> iter:
        return self.data.itertuples(name=self.level)

    def __len__(self) -> int:
        return len(self.data.index)

    def __getitem__(self, key: Union[str, int]) -> dict:
        return self.data.loc[key].to_dict()

    def __contains__(self, key: Union[str, int]) -> bool:
        return key in self.data.index

    def __getattr__(self, key: Union[str, int]) -> dict:
        return self.data.loc[key, "properties"].to_dict()

    def __setattr__(self, key: Union[str, int], value: dict) -> None:
        self.data.loc[key, "properties"] = value


# This class is shown to demonstrate the flexibility of making PropertyStore an abstract class
class DictPropertyStore(PropertyStore):
    @property
    def properties(self) -> dict:
        pass

    def get_properties(self, key) -> dict[Any, Any]:
        pass

    def get_property(self, key, prop_name) -> Any:
        pass

    def set_property(self, key, prop_name, prop_val) -> None:
        pass
