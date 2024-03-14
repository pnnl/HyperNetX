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


# use case: one big dataframe
# types of dataframe
# incidence dataframe, nodes are indexes, edges are column


def fromCombinedDataFrame(
    df: DataFrame,
) -> Tuple[PropertyStore, PropertyStore]:
    """Parse dataframe and create PropertyStore instances

    Returns:

    PropertyStore: edgeids map to edge properties
    PropertyStore: nodeids map to node properties

    Example for edgeids:

    id: property
    Hufflepuff: {housesize: 42}

    example for nodeid

    Half-Blood: {eye_color: variable, hair_color: variable}

    example


    Step 1: determine what type of dataframe (IP, node, edge)
    copy the entityset helper method

    Step2: create the PropertyStore based

    """
    ...
