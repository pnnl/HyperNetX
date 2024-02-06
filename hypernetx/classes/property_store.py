from abc import ABC
from typing import Mapping, Iterable, Union

import pandas as pd


class PropertyStore(ABC):
    def __init__(self, level: Union[str, int]):
        self.level = level
        super().__init__()

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


class DataFramePropertyStore(PropertyStore):
    def __init__(self, level: Union[str, int], data: pd.DataFrame):
        """
        :param level:
        :param data: pd.DataFrame This dataframe must have the shape of the following:

        id          | uid   | weight            | properties
        int or str  | str    | int, default = 1 | dictionary
        """
        super.__init__(level)
        self.data = data

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


class DictPropertyStore(PropertyStore):
    def __init__(
        self,
        level: Union[str, int],
        data: Mapping[Union[str, int], Iterable[Union[str, int]]],
    ):
        super.__init__(level)
        self.data = data
