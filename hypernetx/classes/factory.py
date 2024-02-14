from typing import Any, Tuple, Sequence
from hypernetx.classes.property_store import PropertyStore
from hypernetx.classes.incidence_store import IncidenceStore

import pandas as pd

# HNX
# ability to lookup properties

# return a loc/ series
# set a attribute into dictionary
# key, value pair

# setattr affects the class attributes
# get item, set item (
# id, key, value

# remove getattr, setattr
# setattr(id, key, val)
# use MultiIndex instead of tuple for Id
# index based on edgeid, nodeid, multiindex (no dupes)


def create_property_store_df(
    data: pd.DataFrame, properties: pd.DataFrame, data_cols: Sequence[str | int]
) -> Tuple[PropertyStore, PropertyStore, PropertyStore, IncidenceStore]:
    """Parse data and properties and create PropertyStore instances

    Returns:

    """

    # create three dataframes (no multiindex) - nodes, edge, incidence pairs,
    # 4 helper methods
    # create df for nodes
    # create df for edge
    # create df for incidence pairs
    # create df for incidence store

    # By default: create properties even if not provided
    # for laterenable properties flag; set default true

    # step 1: create initial dfp

    # step 2:
    # TODO: Fix to check the shape of properties or redo properties format
    column_map = {
        old: new
        for old, new in zip(
            (level_col, id_col, misc_col),
            (*self.properties.index.names, self._misc_props_col),
        )
        if old is not None
    }
    props = props.rename(columns=column_map)
    props = props.rename_axis(index=column_map)

    # Step 3:
    # create the properties from dataframe

    # names of property table idx-levels for level and item id, respectively
    # ``item`` used instead of ``id`` to avoid redefining python built-in func `id`
    level, item = self.properties.index.names
    if props.index.nlevels > 1:  # props has MultiIndex
        # drop all idx-levels from props other than level and id (if present)
        extra_levels = [
            idx_lev for idx_lev in props.index.names if idx_lev not in (level, item)
        ]
        props = props.reset_index(level=extra_levels)

    try:
        # if props index is already in the correct format,
        # enforce the correct idx-level ordering
        props.index = props.index.reorder_levels((level, item))
    except AttributeError:  # props is not in (level, id) MultiIndex format
        # if the index matches level or id, drop index to column
        if props.index.name in (level, item):
            props = props.reset_index()
        index_cols = [item]
        if level in props:
            index_cols.insert(0, level)
        try:
            props = props.set_index(index_cols, verify_integrity=True)
        except ValueError:
            warnings.warn(
                "duplicate (level, ID) rows will be dropped after first occurrence"
            )
            props = props.drop_duplicates(index_cols)
            props = props.set_index(index_cols)

    if self._misc_props_col in props:
        try:
            props[self._misc_props_col] = props[self._misc_props_col].apply(
                literal_eval
            )
        except ValueError:
            pass  # data already parsed, no literal eval needed
        else:
            warnings.warn("parsed property dict column from string literal")

    if props.index.nlevels == 1:
        props = props.reindex(self.properties.index, level=1)

    # combine with existing properties
    # non-null values in new props override existing value
    properties = props.combine_first(self.properties)
    # update misc. column to combine existing and new misc. property dicts
    # new props override existing value for overlapping misc. property dict keys
    properties[self._misc_props_col] = self.properties[self._misc_props_col].combine(
        properties[self._misc_props_col],
        lambda x, y: {**(x if pd.notna(x) else {}), **(y if pd.notna(y) else {})},
        fill_value={},
    )
    self._properties = properties.sort_index()

    # Step 4:
# def create_property_store_dict(
#     data: pd.DataFrame, properties: dict[int, dict[str | int, dict[Any, Any]]]
# ) -> Tuple[PropertyStore, PropertyStore, PropertyStore]:
#     """Parse data and properties and create PropertyStore instances
#
#     Returns:
#
#     Tuple of three PropertyStores
#     """
#     ...


def _initialize_properties(data: pd.DataFrame, data_cols: Sequence[str | int]):
    # also need to create a data_cols?

    data_cols_final = _create_data_cols(data, data_cols)
    item_levels = [
        (level, item)
        for level, col in enumerate(self._data_cols)
        for item in self.dataframe[col].cat.categories
    ]
    index = pd.MultiIndex.from_tuples(item_levels, names=[level_col, id_col])
    data = [(i, 1, {}) for i in range(len(index))]

    # create the initial properties as a Dataframe using data, index, an
    return pd.DataFrame(
        data=data, index=index, columns=["uid", "weight", self._misc_props_col]
    ).sort_index()


def _create_data_cols(data: pd.DataFrame, data_cols: Sequence[str | int]):
    """store a list of columns that hold entity data (not properties or weights)"""
    # import ipdb; ipdb.set_trace()
    _data_cols = []
    if not data.empty:
        for col in data_cols:
            if isinstance(col, int):
                _data_cols.append(data.columns[col])
            else:
                _data_cols.append(col)
    return data_cols
