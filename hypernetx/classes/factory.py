from typing import Union, Any
from hypernetx.classes.property_store import PropertyStore

import pandas as pd


def create_property_store_df(
    data: pd.DataFrame, level: Union[int, str]
) -> PropertyStore:
    """Parse dataframe and create PropertyStore instances

    Returns:

    PropertyStore: id's map to properties
    """
    ...


def create_property_store_dict(
    data: dict[Union[int, str], dict[Any, Any]]
) -> PropertyStore:
    """Parse dictionary and create PropertyStore instance

    Returns:

    PropertyStore: id's map to properties
    """
    ...


def create_property_store_incidence_pairs_df(data: pd.DataFrame) -> PropertyStore:
    """Parse dataframe and create PropertyStore instance for incidence pairs

    Returns:

    PropertyStore: id's map to properties
    """
    ...


def create_property_store_incidence_pairs_dict(
    data: (
        dict[Union[int, str], dict[Any, Any]]
        | dict[Union[int, str], dict[Union[int, str], dict[Any, Any]]]
    )
) -> PropertyStore:
    """Parse dataframe and create PropertyStore instance for incidence pairs

    Returns:

    PropertyStore: id's map to properties
    """
    ...


def _props2dict(df=None):
    if df is None:
        return {}
    elif isinstance(df, pd.DataFrame):
        return df.set_index(df.columns[0]).to_dict(orient="index")
    else:
        return dict(df)
