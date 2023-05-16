from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
from collections import UserList
from collections.abc import Hashable, Iterable
from pandas.api.types import CategoricalDtype
from ast import literal_eval

from hypernetx.classes.entity import *


class AttrList(UserList):
    """Custom list wrapper for integrated property storage in :class:`Entity`

    Parameters
    ----------
    entity : hypernetx.Entity
    key : tuple of (int, str or int)
        ``(level, item)``
    initlist : list, optional
        list of elements, passed to ``UserList`` constructor
    """

    def __init__(
        self,
        entity: Entity,
        key: tuple[int, str | int],
        initlist: Optional[list] = None,
    ):
        self._entity = entity
        self._key = key
        super().__init__(initlist)

    def __getattr__(self, attr: str) -> Any:
        """Get attribute value from properties of :attr:`entity`

        Parameters
        ----------
        attr : str

        Returns
        -------
        any
            attribute value; None if not found
        """
        if attr == "uidset":
            return frozenset(self.data)
        if attr in ["memberships", "elements"]:
            return self._entity.__getattribute__(attr).get(self._key[1])
        return self._entity.get_property(self._key[1], attr, self._key[0])

    def __setattr__(self, attr: str, val: Any) -> None:
        """Set attribute value in properties of :attr:`entity`

        Parameters
        ----------
        attr : str
        val : any
        """
        if attr in ["_entity", "_key", "data"]:
            object.__setattr__(self, attr, val)
        else:
            self._entity.set_property(self._key[1], attr, val, level=self._key[0])


def encode(data: pd.DataFrame):
    """
    Encode dataframe to numpy array

    Parameters
    ----------
    data : dataframe

    Returns
    -------
    numpy.array

    """
    encoded_array = data.apply(lambda x: x.cat.codes).to_numpy()
    return encoded_array


def assign_weights(df, weights=1, weight_col="cell_weights"):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to assign a weight column to
    weights : array-like or Hashable, optional
        If numpy.ndarray with the same length as df, create a new weight column with
        these values.
        If Hashable, must be the name of a column of df to assign as the weight column
        Otherwise, create a new weight column assigning a weight of 1 to every row
    weight_col : Hashable
        Name for new column if one is created (not used if the name of an existing
        column is passed as weights)

    Returns
    -------
    df : pandas.DataFrame
        The original DataFrame with a new column added if needed
    weight_col : str
        Name of the column assigned to hold weights

    Note
    ----
    TODO: move logic for default weights inside this method
    """

    if isinstance(weights, (list, np.ndarray)):
        df[weight_col] = weights
    else:
        if not weight_col in df:
            df[weight_col] = weights
    # import ipdb; ipdb.set_trace()
    return df, weight_col


def create_properties(
    props: pd.DataFrame
    | dict[str | int, Iterable[str | int]]
    | dict[str | int, dict[str | int, dict[Any, Any]]]
    | None,
    index_cols: list[str],
    misc_col: str,
) -> pd.DataFrame:
    """Helper function for initializing properties and cell properties

    Parameters
    ----------
    props : pandas.DataFrame, dict of iterables, doubly-nested dict, or None
        See documentation of the `properties` parameter in :class:`Entity`,
        `cell_properties` parameter in :class:`EntitySet`
    index_cols : list of str
        names of columns to be used as levels of the MultiIndex
    misc_col : str
        name of column to be used for miscellaneous property dicts

    Returns
    -------
    pandas.DataFrame
        with ``MultiIndex`` on `index_cols`;
        each entry of the miscellaneous column holds dict of
        ``{property name: property value}``
    """

    if isinstance(props, pd.DataFrame) and not props.empty:
        try:
            data = props.set_index(index_cols, verify_integrity=True)
        except ValueError:
            warnings.warn(
                "duplicate (level, ID) rows will be dropped after first occurrence"
            )
            props = props.drop_duplicates(index_cols)
            data = props.set_index(index_cols)

        if misc_col not in data:
            data[misc_col] = [{} for _ in range(len(data))]
        try:
            data[misc_col] = data[misc_col].apply(literal_eval)
        except ValueError:
            pass  # data already parsed, no literal eval needed
        else:
            warnings.warn("parsed property dict column from string literal")

        return data.sort_index()

    # build MultiIndex from dict of {level: iterable of items}
    try:
        item_levels = [(level, item) for level in props for item in props[level]]
        index = pd.MultiIndex.from_tuples(item_levels, names=index_cols)
    # empty MultiIndex if props is None or other unexpected type
    except TypeError:
        index = pd.MultiIndex(levels=([], []), codes=([], []), names=index_cols)

    # get inner data from doubly-nested dict of {level: {item: {prop: val}}}
    try:
        data = [props[level][item] for level, item in index]
    # empty prop dict for each (level, ID) if iterable of items is not a dict
    except (TypeError, IndexError):
        data = [{} for _ in index]

    return pd.DataFrame({misc_col: data}, index=index).sort_index()


def remove_row_duplicates(
    df, data_cols, weights=1, weight_col="cell_weights", aggregateby=None
):
    """
    Removes and aggregates duplicate rows of a DataFrame using groupby

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to remove or aggregate duplicate rows from
    data_cols : list
        A list of column names in df to perform the groupby on / remove duplicates from
    weights : array-like or Hashable, optional
        Argument passed to assign_weights
    aggregateby : str, optional, default='sum'
        A valid aggregation method for pandas groupby
        If None, drop duplicates without aggregating weights

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with duplicate rows removed or aggregated
    weight_col : Hashable
        The name of the column holding aggregated weights, or None if aggregateby=None
    """
    df = df.copy()
    categories = {}
    for col in data_cols:
        if df[col].dtype.name == "category":
            categories[col] = df[col].cat.categories
            df[col] = df[col].astype(categories[col].dtype)
    df, weight_col = assign_weights(
        df, weights=weights, weight_col=weight_col
    )  ### reconcile this with defaults weights.
    if not aggregateby:
        df = df.drop_duplicates(subset=data_cols)
        df[data_cols] = df[data_cols].astype("category")
        return df, weight_col

    else:
        aggby = {col: "first" for col in df.columns}
        if isinstance(aggregateby, str):
            aggby[weight_col] = aggregateby
        else:
            aggby.update(aggregateby)
        # import ipdb; ipdb.set_trace(context=8)
        df = df.groupby(data_cols, as_index=False, sort=False).agg(aggby)

        # for col in categories:
        # df[col] = df[col].astype(CategoricalDtype(categories=categories[col]))
        df[data_cols] = df[data_cols].astype("category")

    return df, weight_col


# https://stackoverflow.com/a/7205107
def merge_nested_dicts(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_nested_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                warnings.warn(
                    f'Conflict at {",".join(path + [str(key)])}, keys ignored'
                )
        else:
            a[key] = b[key]
    return a


## https://www.geeksforgeeks.org/python-find-depth-of-a-dictionary/
def dict_depth(dic, level=0):
    ### checks if there is a nested dict, quits once level > 2
    if level > 2:
        return level
    if not isinstance(dic, dict) or not dic:
        return level
    return min(dict_depth(dic[key], level + 1) for key in dic)
