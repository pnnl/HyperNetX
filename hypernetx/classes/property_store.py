from abc import ABC, abstractmethod

from pandas import DataFrame

from hypernetx.classes.helpers import T, AttrList
from typing import Mapping, Iterable, Tuple


class PropertyStore(ABC):
    def __init__(self, level: T):
        self.level = level
        super().__init__()

    @abstractmethod
    def uidset(self) -> set:
        """Labels of all items in the underlying data table

        Returns
        -------
        set
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """The number of items in the underlying data table

        Returns
        -------
        int
        """
        pass

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

    def __getitem__(self, key: T) -> dict:
        """Returns the common attributes (e.g. weight) and properties of a key in the underlying data table

        Parameters
        ----------
        key : str | int

        Returns
        -------
        dict
        """
        ...

    def __getattr__(self, key: T) -> dict:
        """Returns the properties of a key in the underlying data table

        Parameters
        ----------
        key : str | int

        Returns
        -------
        dict

        """
        ...

    def __setattr__(self, key: T, value: dict) -> None:
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

    def __contains__(self, key: T) -> bool:
        """Returns true if key is in the underlying data table; false otherwise

        Parameters
        ----------
        key : str | int

        Returns
        -------
        bool
        """
        ...


class DictPropertyStore(PropertyStore):
    def __init__(self, level: T, data: Mapping[T, Iterable[T]]):
        super.__init__(level)
        self.data = data

    def uidset(self):
        ...

    def size(self):
        ...

    # TODO: override magic methods defined in abstract class


class DataFramePropertyStore(PropertyStore):
    def __init__(self, level: T, data: DataFrame):
        super.__init__(level)
        self.data = data

    def uidset(self):
        ...

    def size(self):
        ...

    # TODO: override magic methods defined in abstract class


#####################################################################################################
# Factory Methods
#####################################################################################################

# Notes
# Hypergraph class allows 5 types of setsystems which are used to create the EntitySet
# Eventually there five types will be used to create the PropertyStore
# Iterable of Iterables, Dict of Iterables, Dict of Dict, pandas.Dataframe, numpy.ndarray


class PropertyStoreFactory:
    @staticmethod
    def from_dataframe(
        level: T, data: DataFrame | Mapping[T, Mapping[T, T]]
    ) -> PropertyStore:
        """Create a PropertyStore instance based on level and data

        Process dataframe into valid input which will be used to create a PropertyStore instance

        Ideally the dataframe should be an incident matrix of all the edge-node pairs of the hypergraph.
        The first column should be the edges, second column should be the nodes
        The properties column should be labeled 'properties'
        All other columns are considered cell attributes
        """
        # do some processing on data
        ...
        return DataFramePropertyStore(0, data)

    @staticmethod
    def from_dict(level: T, data: Mapping[T, Mapping[T, T]]) -> PropertyStore:
        """Create a PropertyStore instance based on level and data

        Process into valid input which will be used to create a PropertyStore instance
        """
        # do some processing on data
        ...
        return DictPropertyStore(0, data)
