import numpy as np
import pandas as pd
from collections import UserList
from collections.abc import Hashable
from pandas.api.types import CategoricalDtype

class AttrList(UserList):
    def __init__(self,entity,key,initlist=None):
        self._entity = entity
        self._key = key
        super().__init__(initlist)

    def __getattr__(self,attr):
        return self._entity.properties[self._key].squeeze().get(attr)

    def __setattr__(self,attr,val):
        if attr in ['_entity','_key','data']:
            object.__setattr__(self,attr,val)
        else:
            keyprops = self._entity.properties.get(self._key)
            if keyprops is not None:
                keyprops = keyprops.squeeze()
                keyprops.update({attr: val})
            else:
                keyprops = {attr: val}
            self._entity.properties[self._key] = keyprops

def assign_weights(df, weights=None, weight_col="cell_weights"):
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
    """
    if isinstance(weights, (list, np.ndarray)) and len(weights) == len(df):
        df[weight_col] = weights
    elif isinstance(weights, Hashable) and weights in df:
        weight_col = weights
    else:
        df[weight_col] = np.ones(len(df), dtype=int)

    return df, weight_col

def update_properties(props, new_props):
    if new_props is None:
        return props
    update = props.index.intersection(new_props.index)
    for idx in update:
        props[idx].update(new_props[idx])
    new_props = new_props[~new_props.index.isin(update)]
    return pd.concat((props, new_props))


def remove_row_duplicates(df, data_cols, weights=None, aggregateby="sum"):
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

    if not aggregateby:
        df = df.drop_duplicates(subset=data_cols)

    df, weight_col = assign_weights(df, weights=weights)

    if aggregateby:
        df = df.groupby(data_cols, as_index=False, sort=False).agg(
            {weight_col: aggregateby}
        )

    for col in categories:
        df[col] = df[col].astype(CategoricalDtype(categories=categories[col]))

    return df, weight_col