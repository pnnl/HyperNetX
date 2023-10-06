from __future__ import annotations

import warnings
from ast import literal_eval
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Mapping, Sequence, Iterable
from typing import Union, TypeVar, Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hypernetx.classes.helpers import (
    AttrList,
    assign_weights,
    remove_row_duplicates,
    dict_depth,
)

T = TypeVar("T", bound=Union[str, int])


class Entity:
    """Base class for handling N-dimensional data when building network-like models,
    i.e., :class:`Hypergraph`

    Parameters
    ----------
    entity : pandas.DataFrame, dict of lists or sets, list of lists or sets, optional
        If a ``DataFrame`` with N columns,
        represents N-dimensional entity data (data table).
        Otherwise, represents 2-dimensional entity data (system of sets).
        TODO: Test for compatibility with list of Entities and update docs
    data : numpy.ndarray, optional
        2D M x N ``ndarray`` of ``ints`` (data table);
        sparse representation of an N-dimensional incidence tensor with M nonzero cells.
        Ignored if `entity` is provided.
    static : bool, default=True
        If ``True``, entity data may not be altered,
        and the :attr:`state_dict <_state_dict>` will never be cleared.
        Otherwise, rows may be added to and removed from the data table,
        and updates will clear the :attr:`state_dict <_state_dict>`.
    labels : collections.OrderedDict of lists, optional
        User-specified labels in corresponding order to ``ints`` in `data`.
        Ignored if `entity` is provided or `data` is not provided.
    uid : hashable, optional
        A unique identifier for the object
    weights : str or sequence of float, optional
        User-specified cell weights corresponding to entity data.
        If sequence of ``floats`` and `entity` or `data` defines a data table,
            length must equal the number of rows.
        If sequence of ``floats`` and `entity` defines a system of sets,
            length must equal the total sum of the sizes of all sets.
        If ``str`` and `entity` is a ``DataFrame``,
            must be the name of a column in `entity`.
        Otherwise, weight for all cells is assumed to be 1.
    aggregateby : {'sum', 'last', count', 'mean','median', max', 'min', 'first', None}
        Name of function to use for aggregating cell weights of duplicate rows when
        `entity` or `data` defines a data table, default is "sum".
        If None, duplicate rows will be dropped without aggregating cell weights.
        Effectively ignored if `entity` defines a system of sets.
    properties : pandas.DataFrame or doubly-nested dict, optional
        User-specified properties to be assigned to individual items in the data, i.e.,
        cell entries in a data table; sets or set elements in a system of sets.
        See Notes for detailed explanation.
        If ``DataFrame``, each row gives
        ``[optional item level, item label, optional named properties,
        {property name: property value}]``
        (order of columns does not matter; see note for an example).
        If doubly-nested dict,
        ``{item level: {item label: {property name: property value}}}``.
    misc_props_col, level_col, id_col : str, default="properties", "level, "id"
        Column names for miscellaneous properties, level index, and item name in
        :attr:`properties`; see Notes for explanation.

    Notes
    -----
    A property is a named attribute assigned to a single item in the data.

    You can pass a **table of properties** to `properties` as a ``DataFrame``:

    +------------+---------+----------------+-------+------------------+
    | Level      | ID      | [explicit      | [...] | misc. properties |
    | (optional) |         | property type] |       |                  |
    +============+=========+================+=======+==================+
    | 0          | level 0 | property value | ...   | {property name:  |
    |            | item    |                |       | property value}  |
    +------------+---------+----------------+-------+------------------+
    | 1          | level 1 | property value | ...   | {property name:  |
    |            | item    |                |       | property value}  |
    +------------+---------+----------------+-------+------------------+
    | ...        | ...     | ...            | ...   | ...              |
    +------------+---------+----------------+-------+------------------+
    | N          | level N | property value | ...   | {property name:  |
    |            | item    |                |       | property value}  |
    +------------+---------+----------------+-------+------------------+

    The Level column is optional. If not provided, properties will be assigned by ID
    (i.e., if an ID appears at multiple levels, the same properties will be assigned to
    all occurrences).

    The names of the Level (if provided) and ID columns must be specified by `level_col`
    and `id_col`. `misc_props_col` can be used to specify the name of the column to be used
    for miscellaneous properties; if no column by that name is found,
    a new column will be created and populated with empty ``dicts``.
    All other columns will be considered explicit property types.
    The order of the columns does not matter.

    This method assumes that there are no rows with the same (Level, ID);
    if duplicates are found, all but the first occurrence will be dropped.

    """

    def __init__(
        self,
        entity: Optional[
            pd.DataFrame | Mapping[T, Iterable[T]] | Iterable[Iterable[T]]
        ] = None,
        data_cols: Sequence[T] = [0, 1],
        data: Optional[np.ndarray] = None,
        static: bool = False,
        labels: Optional[OrderedDict[T, Sequence[T]]] = None,
        uid: Optional[Hashable] = None,
        weight_col: Optional[str | int] = "cell_weights",
        weights: Optional[Sequence[float] | float | int | str] = 1,
        aggregateby: Optional[str | dict] = "sum",
        properties: Optional[pd.DataFrame | dict[int, dict[T, dict[Any, Any]]]] = None,
        misc_props_col: str = "properties",
        level_col: str = "level",
        id_col: str = "id",
    ):
        # set unique identifier
        self._uid = uid or None

        # if static, the original data cannot be altered
        # the state dict stores all computed values that may need to be updated
        # if the data is altered - the dict will be cleared when data is added
        # or removed
        self._static = static
        self._state_dict = {}

        # entity data is stored in a DataFrame for basic access without the
        # need for any label encoding lookups
        if isinstance(entity, pd.DataFrame):
            self._dataframe = entity.copy()

        # if the entity data is passed as a dict of lists or a list of lists,
        # we convert it to a 2-column dataframe by exploding each list to cover
        # one row per element for a dict of lists, the first level/column will
        # be filled in with dict keys for a list of N lists, 0,1,...,N will be
        # used to fill the first level/column
        elif isinstance(entity, (dict, list)):
            # convert dict of lists to 2-column dataframe
            entity = pd.Series(entity).explode()
            self._dataframe = pd.DataFrame(
                {data_cols[0]: entity.index.to_list(), data_cols[1]: entity.values}
            )

        # if a 2d numpy ndarray is passed, store it as both a DataFrame and an
        # ndarray in the state dict
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self._state_dict["data"] = data
            self._dataframe = pd.DataFrame(data)
            # if a dict of labels was passed, use keys as column names in the
            # DataFrame, translate the dataframe, and store the dict of labels
            # in the state dict
            if isinstance(labels, dict) and len(labels) == len(self._dataframe.columns):
                self._dataframe.columns = labels.keys()
                self._state_dict["labels"] = labels

                for col in self._dataframe:
                    self._dataframe[col] = pd.Categorical.from_codes(
                        self._dataframe[col], categories=labels[col]
                    )

        # create an empty Entity
        else:
            self._dataframe = pd.DataFrame()

        # assign a new or existing column of the dataframe to hold cell weights
        self._dataframe, self._cell_weight_col = assign_weights(
            self._dataframe, weights=weights, weight_col=weight_col
        )
        # import ipdb; ipdb.set_trace()
        # store a list of columns that hold entity data (not properties or
        # weights)
        # self._data_cols = list(self._dataframe.columns.drop(self._cell_weight_col))
        self._data_cols = []
        for col in data_cols:
            # TODO: default arguments fail for empty Entity; data_cols has two elements but _dataframe has only one element
            if isinstance(col, int):
                self._data_cols.append(self._dataframe.columns[col])
            else:
                self._data_cols.append(col)

        # each entity data column represents one dimension of the data
        # (data updates can only add or remove rows, so this isn't stored in
        # state dict)
        self._dimsize = len(self._data_cols)

        # remove duplicate rows and aggregate cell weights as needed
        # import ipdb; ipdb.set_trace()
        self._dataframe, _ = remove_row_duplicates(
            self._dataframe,
            self._data_cols,
            weight_col=self._cell_weight_col,
            aggregateby=aggregateby,
        )

        # set the dtype of entity data columns to categorical (simplifies
        # encoding, etc.)
        ### This is automatically done in remove_row_duplicates
        # self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
        #     "category"
        # )

        # create properties
        item_levels = [
            (level, item)
            for level, col in enumerate(self._data_cols)
            for item in self.dataframe[col].cat.categories
        ]
        index = pd.MultiIndex.from_tuples(item_levels, names=[level_col, id_col])
        data = [(i, 1, {}) for i in range(len(index))]
        self._properties = pd.DataFrame(
            data=data, index=index, columns=["uid", "weight", misc_props_col]
        ).sort_index()
        self._misc_props_col = misc_props_col
        if properties is not None:
            self.assign_properties(properties)

    @property
    def data(self):
        """Sparse representation of the data table as an incidence tensor

        This can also be thought of as an encoding of `dataframe`, where items in each column of
        the data table are translated to their int position in the `self.labels[column]` list
        Returns
        -------
        numpy.ndarray
            2D array of ints representing rows of the underlying data table as indices in an incidence tensor

        See Also
        --------
        labels, dataframe

        """
        # generate if not already stored in state dict
        if "data" not in self._state_dict:
            if self.empty:
                self._state_dict["data"] = np.zeros((0, 0), dtype=int)
            else:
                # assumes dtype of data cols is already converted to categorical
                # and state dict has been properly cleared after updates
                self._state_dict["data"] = (
                    self._dataframe[self._data_cols]
                    .apply(lambda x: x.cat.codes)
                    .to_numpy()
                )

        return self._state_dict["data"]

    @property
    def labels(self):
        """Labels of all items in each column of the underlying data table

        Returns
        -------
        dict of lists
            dict of {column name: [item labels]}
            The order of [item labels] corresponds to the int encoding of each item in `self.data`.

        See Also
        --------
        data, dataframe
        """
        # generate if not already stored in state dict
        if "labels" not in self._state_dict:
            # assumes dtype of data cols is already converted to categorical
            # and state dict has been properly cleared after updates
            self._state_dict["labels"] = {
                col: self._dataframe[col].cat.categories.to_list()
                for col in self._data_cols
            }

        return self._state_dict["labels"]

    @property
    def cell_weights(self):
        """Cell weights corresponding to each row of the underlying data table

        Returns
        -------
        dict of {tuple: int or float}
            Keyed by row of data table (as a tuple)
        """
        # generate if not already stored in state dict
        if "cell_weights" not in self._state_dict:
            if self.empty:
                self._state_dict["cell_weights"] = {}
            else:
                self._state_dict["cell_weights"] = self._dataframe.set_index(
                    self._data_cols
                )[self._cell_weight_col].to_dict()

        return self._state_dict["cell_weights"]

    @property
    def dimensions(self):
        """Dimensions of data i.e., the number of distinct items in each level (column) of the underlying data table

        Returns
        -------
        tuple of ints
            Length and order corresponds to columns of `self.dataframe` (excluding cell weight column)
        """
        # generate if not already stored in state dict
        if "dimensions" not in self._state_dict:
            if self.empty:
                self._state_dict["dimensions"] = tuple()
            else:
                self._state_dict["dimensions"] = tuple(
                    self._dataframe[self._data_cols].nunique()
                )

        return self._state_dict["dimensions"]

    @property
    def dimsize(self):
        """Number of levels (columns) in the underlying data table

        Returns
        -------
        int
            Equal to length of `self.dimensions`
        """
        return self._dimsize

    @property
    def properties(self) -> pd.DataFrame:
        # Dev Note: Not sure what this contains, when running tests it contained an empty pandas series
        """Properties assigned to items in the underlying data table

        Returns
        -------
        pandas.DataFrame
        """

        return self._properties

    @property
    def uid(self):
        # Dev Note: This also returned nothing in my harry potter dataset, not sure if it was supposed to contain anything
        """User-defined unique identifier for the `Entity`

        Returns
        -------
        hashable
        """
        return self._uid

    @property
    def uidset(self):
        """Labels of all items in level 0 (first column) of the underlying data table

        Returns
        -------
        frozenset

        See Also
        --------
        children : Labels of all items in level 1 (second column)
        uidset_by_level, uidset_by_column :
            Labels of all items in any level (column); specified by level index or column name
        """
        return self.uidset_by_level(0)

    @property
    def children(self):
        """Labels of all items in level 1 (second column) of the underlying data table

        Returns
        -------
        frozenset

        See Also
        --------
        uidset : Labels of all items in level 0 (first column)
        uidset_by_level, uidset_by_column :
            Labels of all items in any level (column); specified by level index or column name
        """
        return self.uidset_by_level(1)

    def uidset_by_level(self, level):
        """Labels of all items in a particular level (column) of the underlying data table

        Parameters
        ----------
        level : int

        Returns
        -------
        frozenset

        See Also
        --------
        uidset : Labels of all items in level 0 (first column)
        children : Labels of all items in level 1 (second column)
        uidset_by_column : Same functionality, takes the column name instead of level index
        """
        if self.is_empty(level):
            return {}
        col = self._data_cols[level]
        return self.uidset_by_column(col)

    def uidset_by_column(self, column):
        # Dev Note: This threw an error when trying it on the harry potter dataset,
        # when trying 0, or 1 for column. I'm not sure how this should be used
        """Labels of all items in a particular column (level) of the underlying data table

        Parameters
        ----------
        column : Hashable
            Name of a column in `self.dataframe`

        Returns
        -------
        frozenset

        See Also
        --------
        uidset : Labels of all items in level 0 (first column)
        children : Labels of all items in level 1 (second column)
        uidset_by_level : Same functionality, takes the level index instead of column name
        """
        # generate if not already stored in state dict
        if "uidset" not in self._state_dict:
            self._state_dict["uidset"] = {}
        if column not in self._state_dict["uidset"]:
            self._state_dict["uidset"][column] = set(
                self._dataframe[column].dropna().unique()
            )

        return self._state_dict["uidset"][column]

    @property
    def elements(self):
        """System of sets representation of the first two levels (columns) of the underlying data table

        Each item in level 0 (first column) defines a set containing all the level 1
        (second column) items with which it appears in the same row of the underlying
        data table

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {level 0 item : AttrList(level 1 items)}

        See Also
        --------
        incidence_dict : same data as dict of list
        memberships :
            dual of this representation,
            i.e., each item in level 1 (second column) defines a set
        elements_by_level, elements_by_column :
            system of sets representation of any two levels (columns); specified by level index or column name

        """
        if self._dimsize < 2:
            return {k: AttrList(entity=self, key=(0, k)) for k in self.uidset}

        return self.elements_by_level(0, 1)

    @property
    def incidence_dict(self) -> dict[T, list[T]]:
        """System of sets representation of the first two levels (columns) of the underlying data table

        Returns
        -------
        dict of list
            System of sets representation as dict of {level 0 item : AttrList(level 1 items)}

        See Also
        --------
        elements : same data as dict of AttrList

        """
        return {item: elements.data for item, elements in self.elements.items()}

    @property
    def memberships(self):
        """System of sets representation of the first two levels (columns) of the
        underlying data table

        Each item in level 1 (second column) defines a set containing all the level 0
        (first column) items with which it appears in the same row of the underlying
        data table

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {level 1 item : AttrList(level 0 items)}

        See Also
        --------
        elements : dual of this representation i.e., each item in level 0 (first column) defines a set
        elements_by_level, elements_by_column :
            system of sets representation of any two levels (columns); specified by level index or column name

        """

        return self.elements_by_level(1, 0)

    def elements_by_level(self, level1, level2):
        """System of sets representation of two levels (columns) of the underlying data table

        Each item in level1 defines a set containing all the level2 items
        with which it appears in the same row of the underlying data table

        Properties can be accessed and assigned to items in level1

        Parameters
        ----------
        level1 : int
            index of level whose items define sets
        level2 : int
            index of level whose items are elements in the system of sets

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {level1 item : AttrList(level2 items)}

        See Also
        --------
        elements, memberships : dual system of sets representations of the first two levels (columns)
        elements_by_column : same functionality, takes column names instead of level indices

        """
        col1 = self._data_cols[level1]
        col2 = self._data_cols[level2]
        return self.elements_by_column(col1, col2)

    def elements_by_column(self, col1, col2):
        """System of sets representation of two columns (levels) of the underlying data table

        Each item in col1 defines a set containing all the col2 items
        with which it appears in the same row of the underlying data table

        Properties can be accessed and assigned to items in col1

        Parameters
        ----------
        col1 : Hashable
            name of column whose items define sets
        col2 : Hashable
            name of column whose items are elements in the system of sets

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {col1 item : AttrList(col2 items)}

        See Also
        --------
        elements, memberships : dual system of sets representations of the first two columns (levels)
        elements_by_level : same functionality, takes level indices instead of column names

        """
        if "elements" not in self._state_dict:
            self._state_dict["elements"] = defaultdict(dict)
        if col2 not in self._state_dict["elements"][col1]:
            level = self.index(col1)
            elements = (
                self._dataframe.groupby(col1, observed=False)[col2].unique().to_dict()
            )
            self._state_dict["elements"][col1][col2] = {
                item: AttrList(entity=self, key=(level, item), initlist=elem)
                for item, elem in elements.items()
            }

        return self._state_dict["elements"][col1][col2]

    @property
    def dataframe(self):
        """The underlying data table stored by the Entity

        Returns
        -------
        pandas.DataFrame
        """
        return self._dataframe

    @property
    def isstatic(self):
        # Dev Note: I'm guessing this is no longer necessary?
        """Whether to treat the underlying data as static or not

        If True, the underlying data may not be altered, and the state_dict will never be cleared
        Otherwise, rows may be added to and removed from the data table, and updates will clear the state_dict

        Returns
        -------
        bool
        """
        return self._static

    def size(self, level=0):
        """The number of items in a level of the underlying data table

        Equivalent to ``self.dimensions[level]``

        Parameters
        ----------
        level : int, default=0

        Returns
        -------
        int

        See Also
        --------
        dimensions
        """
        # TODO: Since `level` is not validated, we assume that self.dimensions should be an array large enough to access index `level`
        return self.dimensions[level]

    @property
    def empty(self):
        """Whether the underlying data table is empty or not

        Returns
        -------
        bool

        See Also
        --------
        is_empty : for checking whether a specified level (column) is empty
        dimsize : 0 if empty
        """
        return self._dimsize == 0

    def is_empty(self, level=0):
        """Whether a specified level (column) of the underlying data table is empty or not

        Returns
        -------
        bool

        See Also
        --------
        empty : for checking whether the underlying data table is empty
        size : number of items in a level (columns); 0 if level is empty
        """
        return self.empty or self.size(level) == 0

    def __len__(self):
        """Number of items in level 0 (first column)

        Returns
        -------
        int
        """
        return self.dimensions[0]

    def __contains__(self, item):
        """Whether an item is contained within any level of the data

        Parameters
        ----------
        item : str

        Returns
        -------
        bool
        """
        for labels in self.labels.values():
            if item in labels:
                return True
        return False

    def __getitem__(self, item):
        """Access into the system of sets representation of the first two levels (columns) given by `elements`

        Can be used to access and assign properties to an ``item`` in level 0 (first column)

        Parameters
        ----------
        item : str
            label of an item in level 0 (first column)

        Returns
        -------
        AttrList :
            list of level 1 items in the set defined by ``item``

        See Also
        --------
        uidset, elements
        """
        return self.elements[item]

    def __iter__(self):
        """Iterates over items in level 0 (first column) of the underlying data table

        Returns
        -------
        Iterator

        See Also
        --------
        uidset, elements
        """
        return iter(self.elements)

    def __call__(self, label_index=0):
        # Dev Note (Madelyn) : I don't think this is the intended use of __call__, can we change/deprecate?
        """Iterates over items labels in a specified level (column) of the underlying data table

        Parameters
        ----------
        label_index : int
            level index

        Returns
        -------
        Iterator

        See Also
        --------
        labels
        """
        return iter(self.labels[self._data_cols[label_index]])

    # def __repr__(self):
    #     """String representation of the Entity

    #     e.g., "Entity(uid, [level 0 items], {item: {property name: property value}})"

    #     Returns
    #     -------
    #     str
    #     """
    #     return "hypernetx.classes.entity.Entity"

    # def __str__(self):
    #     return "<class 'hypernetx.classes.entity.Entity'>"

    def index(self, column, value=None):
        """Get level index corresponding to a column and (optionally) the index of a value in that column

        The index of ``value`` is its position in the list given by ``self.labels[column]``, which is used
        in the integer encoding of the data table ``self.data``

        Parameters
        ----------
        column: str
            name of a column in self.dataframe
        value : str, optional
            label of an item in the specified column

        Returns
        -------
        int or (int, int)
            level index corresponding to column, index of value if provided

        See Also
        --------
        indices : for finding indices of multiple values in a column
        level : same functionality, search for the value without specifying column
        """
        if "keyindex" not in self._state_dict:
            self._state_dict["keyindex"] = {}
        if column not in self._state_dict["keyindex"]:
            self._state_dict["keyindex"][column] = self._dataframe[
                self._data_cols
            ].columns.get_loc(column)

        if value is None:
            return self._state_dict["keyindex"][column]

        if "index" not in self._state_dict:
            self._state_dict["index"] = defaultdict(dict)
        if value not in self._state_dict["index"][column]:
            self._state_dict["index"][column][value] = self._dataframe[
                column
            ].cat.categories.get_loc(value)

        return (
            self._state_dict["keyindex"][column],
            self._state_dict["index"][column][value],
        )

    def indices(self, column, values):
        """Get indices of one or more value(s) in a column

        Parameters
        ----------
        column : str
        values : str or iterable of str

        Returns
        -------
        list of int
            indices of values

        See Also
        --------
        index : for finding level index of a column and index of a single value
        """
        if isinstance(values, Hashable):
            values = [values]

        if "index" not in self._state_dict:
            self._state_dict["index"] = defaultdict(dict)
        for v in values:
            if v not in self._state_dict["index"][column]:
                self._state_dict["index"][column][v] = self._dataframe[
                    column
                ].cat.categories.get_loc(v)

        return [self._state_dict["index"][column][v] for v in values]

    def translate(self, level, index):
        """Given indices of a level and value(s), return the corresponding value label(s)

        Parameters
        ----------
        level : int
            level index
        index : int or list of int
            value index or indices

        Returns
        -------
        str or list of str
            label(s) corresponding to value index or indices

        See Also
        --------
        translate_arr : translate a full row of value indices across all levels (columns)
        """
        column = self._data_cols[level]

        if isinstance(index, (int, np.integer)):
            return self.labels[column][index]

        return [self.labels[column][i] for i in index]

    def translate_arr(self, coords):
        """Translate a full encoded row of the data table e.g., a row of ``self.data``

        Parameters
        ----------
        coords : tuple of ints
            encoded value indices, with one value index for each level of the data

        Returns
        -------
        list of str
            full row of translated value labels
        """
        assert len(coords) == self._dimsize
        translation = []
        for level, index in enumerate(coords):
            translation.append(self.translate(level, index))

        return translation

    def level(self, item, min_level=0, max_level=None, return_index=True):
        """First level containing the given item label

        Order of levels corresponds to order of columns in `self.dataframe`

        Parameters
        ----------
        item : str
        min_level, max_level : int, optional
            inclusive bounds on range of levels to search for item
        return_index : bool, default=True
            If True, return index of item within the level

        Returns
        -------
        int, (int, int), or None
            index of first level containing the item, index of item if `return_index=True`
            returns None if item is not found

        See Also
        --------
        index, indices : for finding level and/or value indices when the column is known
        """
        if max_level is None or max_level >= self._dimsize:
            max_level = self._dimsize - 1

        columns = self._data_cols[min_level : max_level + 1]
        levels = range(min_level, max_level + 1)

        for col, lev in zip(columns, levels):
            if item in self.labels[col]:
                if return_index:
                    return self.index(col, item)

                return lev

        print(f'"{item}" not found.')
        return None

    def add(self, *args):
        """Updates the underlying data table with new entity data from multiple sources

        Parameters
        ----------
        *args
            variable length argument list of Entity and/or representations of entity data

        Returns
        -------
        self : Entity

        Warnings
        --------
        Adding an element directly to an Entity will not add the
        element to any Hypergraphs constructed from that Entity, and will cause an error. Use
        :func:`Hypergraph.add_edge <classes.hypergraph.Hypergraph.add_edge>` or
        :func:`Hypergraph.add_node_to_edge <classes.hypergraph.Hypergraph \
            .add_node_to_edge>` instead.

        See Also
        --------
        add_element : update from a single source
        Hypergraph.add_edge, Hypergraph.add_node_to_edge : for adding elements to a Hypergraph

        """
        for item in args:
            self.add_element(item)
        return self

    def add_elements_from(self, arg_set):
        """Adds arguments from an iterable to the data table one at a time

        ..deprecated:: 2.0.0
            Duplicates `add`

        Parameters
        ----------
        arg_set : iterable
            list of Entity and/or representations of entity data

        Returns
        -------
        self : Entity

        """
        for item in arg_set:
            self.add_element(item)
        return self

    def add_element(self, data):
        """Updates the underlying data table with new entity data

        Supports adding from either an existing Entity or a representation of entity
        (data table or labeled system of sets are both supported representations)

        Parameters
        ----------
        data : Entity, `pandas.DataFrame`, or dict of lists or sets
            new entity data

        Returns
        -------
        self : Entity

        Warnings
        --------
        Adding an element directly to an Entity will not add the
        element to any Hypergraphs constructed from that Entity, and will cause an error. Use
        `Hypergraph.add_edge` or `Hypergraph.add_node_to_edge` instead.

        See Also
        --------
        add : takes multiple sources of new entity data as variable length argument list
        Hypergraph.add_edge, Hypergraph.add_node_to_edge : for adding elements to a Hypergraph

        """
        if isinstance(data, Entity):
            df = data.dataframe
            self.__add_from_dataframe(df)

        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
            self.__add_from_dataframe(df)

        if isinstance(data, pd.DataFrame):
            self.__add_from_dataframe(data)

        return self

    def __add_from_dataframe(self, df):
        """Helper function to append rows to `self.dataframe`

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        self : Entity

        """
        if all(col in df for col in self._data_cols):
            new_data = pd.concat((self._dataframe, df), ignore_index=True)
            new_data[self._cell_weight_col] = new_data[self._cell_weight_col].fillna(1)

            self._dataframe, _ = remove_row_duplicates(
                new_data,
                self._data_cols,
                weights=self._cell_weight_col,
            )

            self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
                "category"
            )

            self._state_dict.clear()

    def remove(self, *args):
        """Removes all rows containing specified item(s) from the underlying data table

        Parameters
        ----------
        *args
            variable length argument list of item labels

        Returns
        -------
        self : Entity

        See Also
        --------
        remove_element : remove all rows containing a single specified item

        """
        for item in args:
            self.remove_element(item)
        return self

    def remove_elements_from(self, arg_set):
        """Removes all rows containing specified item(s) from the underlying data table

        ..deprecated: 2.0.0
            Duplicates `remove`

        Parameters
        ----------
        arg_set : iterable
            list of item labels

        Returns
        -------
        self : Entity

        """
        for item in arg_set:
            self.remove_element(item)
        return self

    def remove_element(self, item):
        """Removes all rows containing a specified item from the underlying data table

        Parameters
        ----------
        item
            item label

        Returns
        -------
        self : Entity

        See Also
        --------
        remove : same functionality, accepts variable length argument list of item labels

        """
        updated_dataframe = self._dataframe

        for column in self._dataframe:
            updated_dataframe = updated_dataframe[updated_dataframe[column] != item]

        self._dataframe, _ = remove_row_duplicates(
            updated_dataframe,
            self._data_cols,
            weights=self._cell_weight_col,
        )
        self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
            "category"
        )

        self._state_dict.clear()
        for col in self._data_cols:
            self._dataframe[col] = self._dataframe[col].cat.remove_unused_categories()

    def encode(self, data):
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

    def incidence_matrix(
        self, level1=0, level2=1, weights=False, aggregateby=None, index=False
    ) -> csr_matrix | None:
        """Incidence matrix representation for two levels (columns) of the underlying data table

        If `level1` and `level2` contain N and M distinct items, respectively, the incidence matrix will be M x N.
        In other words, the items in `level1` and `level2` correspond to the columns and rows of the incidence matrix,
        respectively, in the order in which they appear in `self.labels[column1]` and `self.labels[column2]`
        (`column1` and `column2` are the column labels of `level1` and `level2`)

        Parameters
        ----------
        level1 : int, default=0
            index of first level (column)
        level2 : int, default=1
            index of second level
        weights : bool or dict, default=False
            If False all nonzero entries are 1.
            If True all nonzero entries are filled by self.cell_weight
            dictionary values, use :code:`aggregateby` to specify how duplicate
            entries should have weights aggregated.
            If dict of {(level1 item, level2 item): weight value} form;
            only nonzero cells in the incidence matrix will be updated by dictionary,
            i.e., `level1 item` and `level2 item` must appear in the same row at least once in the underlying data table
        aggregateby : {'last', count', 'sum', 'mean','median', max', 'min', 'first', 'last', None}, default='count'
            Method to aggregate weights of duplicate rows in data table.
             If None, then all cell weights will be set to 1.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            sparse representation of incidence matrix (i.e. Compressed Sparse Row matrix)

        Other Parameters
        ----------------
        index : bool, optional
            Not used

        Note
        ----
        In the context of Hypergraphs, think `level1 = edges, level2 = nodes`
        """
        if self.dimsize < 2:
            warnings.warn("Incidence matrix requires two levels of data.")
            return None

        data_cols = [self._data_cols[level2], self._data_cols[level1]]
        weights = self._cell_weight_col if weights else None

        df, weight_col = remove_row_duplicates(
            self._dataframe,
            data_cols,
            weights=weights,
            aggregateby=aggregateby,
        )

        return csr_matrix(
            (df[weight_col], tuple(df[col].cat.codes for col in data_cols))
        )

    def restrict_to_levels(
        self,
        levels: int | Iterable[int],
        weights: bool = False,
        aggregateby: str | None = "sum",
        **kwargs,
    ) -> Entity:
        """Create a new Entity by restricting to a subset of levels (columns) in the
        underlying data table

        Parameters
        ----------
        levels : array-like of int
            indices of a subset of levels (columns) of data
        weights : bool, default=False
            If True, aggregate existing cell weights to get new cell weights
            Otherwise, all new cell weights will be 1
        aggregateby : {'sum', 'first', 'last', 'count', 'mean', 'median', 'max', \
    'min', None}, optional
            Method to aggregate weights of duplicate rows in data table
            If None or `weights`=False then all new cell weights will be 1
        **kwargs
            Extra arguments to `Entity` constructor

        Returns
        -------
        Entity

        Raises
        ------
        KeyError
            If `levels` contains any invalid values

        See Also
        --------
        EntitySet
        """

        levels = np.asarray(levels)
        invalid_levels = (levels < 0) | (levels >= self.dimsize)
        if invalid_levels.any():
            raise KeyError(f"Invalid levels: {levels[invalid_levels]}")

        cols = [self._data_cols[lev] for lev in levels]

        if weights:
            weights = self._cell_weight_col
            cols.append(weights)
            kwargs.update(weights=weights)

        properties = self.properties.loc[levels]
        properties.index = properties.index.remove_unused_levels()
        level_map = {old: new for new, old in enumerate(levels)}
        new_levels = properties.index.levels[0].map(level_map)
        properties.index = properties.index.set_levels(new_levels, level=0)
        level_col, id_col = properties.index.names

        return self.__class__(
            entity=self.dataframe[cols],
            data_cols=cols,
            aggregateby=aggregateby,
            properties=properties,
            misc_props_col=self._misc_props_col,
            level_col=level_col,
            id_col=id_col,
            **kwargs,
        )

    def restrict_to_indices(self, indices, level=0, **kwargs):
        """Create a new Entity by restricting the data table to rows containing specific items in a given level

        Parameters
        ----------
        indices : int or iterable of int
            indices of item label(s) in `level` to restrict to
        level : int, default=0
            level index
        **kwargs
            Extra arguments to `Entity` constructor

        Returns
        -------
        Entity
        """
        column = self._dataframe[self._data_cols[level]]
        values = self.translate(level, indices)
        entity = self._dataframe.loc[column.isin(values)].copy()

        for col in self._data_cols:
            entity[col] = entity[col].cat.remove_unused_categories()
        restricted = self.__class__(
            entity=entity, misc_props_col=self._misc_props_col, **kwargs
        )

        if not self.properties.empty:
            prop_idx = [
                (lv, uid)
                for lv in range(restricted.dimsize)
                for uid in restricted.uidset_by_level(lv)
            ]
            properties = self.properties.loc[prop_idx]
            restricted.assign_properties(properties)
        return restricted

    def assign_properties(
        self,
        props: pd.DataFrame | dict[int, dict[T, dict[Any, Any]]],
        misc_col: Optional[str] = None,
        level_col=0,
        id_col=1,
    ) -> None:
        """Assign new properties to items in the data table, update :attr:`properties`

        Parameters
        ----------
        props : pandas.DataFrame or doubly-nested dict
            See documentation of the `properties` parameter in :class:`Entity`
        level_col, id_col, misc_col : str, optional
            column names corresponding to the levels, items, and misc. properties;
            if None, default to :attr:`_level_col`, :attr:`_id_col`, :attr:`_misc_props_col`,
            respectively.

        See Also
        --------
        properties
        """
        # mapping from user-specified level, id, misc column names to internal names
        ### This will fail if there isn't a level column

        if isinstance(props, pd.DataFrame):
            ### Fix to check the shape of properties or redo properties format
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
            self._properties_from_dataframe(props)

        if isinstance(props, dict):
            ### Expects nested dictionary with keys corresponding to level and id
            self._properties_from_dict(props)

    def _properties_from_dataframe(self, props: pd.DataFrame) -> None:
        """Private handler for updating :attr:`properties` from a DataFrame

        Parameters
        ----------
        props

        Notes
        -----
        For clarity in in-line developer comments:

        idx-level
            refers generally to a level of a MultiIndex
        level
            refers specifically to the idx-level in the MultiIndex of :attr:`properties`
            that stores the level/column id for the item
        """
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
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        properties = props.combine_first(self.properties)
        warnings.simplefilter(action="default", category=RuntimeWarning)
        # update misc. column to combine existing and new misc. property dicts
        # new props override existing value for overlapping misc. property dict keys
        properties[self._misc_props_col] = self.properties[
            self._misc_props_col
        ].combine(
            properties[self._misc_props_col],
            lambda x, y: {**(x if pd.notna(x) else {}), **(y if pd.notna(y) else {})},
            fill_value={},
        )
        self._properties = properties.sort_index()

    def _properties_from_dict(self, props: dict[int, dict[T, dict[Any, Any]]]) -> None:
        """Private handler for updating :attr:`properties` from a doubly-nested dict

        Parameters
        ----------
        props
        """
        # TODO: there may be a more efficient way to convert this to a dataframe instead
        #  of updating one-by-one via nested loop, but checking whether each prop_name
        #  belongs in a designated existing column or the misc. property dict column
        #  makes it more challenging
        #  For now: only use nested loop update if non-misc. columns currently exist
        if len(self.properties.columns) > 1:
            for level in props:
                for item in props[level]:
                    for prop_name, prop_val in props[level][item].items():
                        self.set_property(item, prop_name, prop_val, level)
        else:
            item_keys = pd.MultiIndex.from_tuples(
                [(level, item) for level in props for item in props[level]],
                names=self.properties.index.names,
            )
            props_data = [props[level][item] for level, item in item_keys]
            props = pd.DataFrame({self._misc_props_col: props_data}, index=item_keys)
            self._properties_from_dataframe(props)

    def _property_loc(self, item: T) -> tuple[int, T]:
        """Get index in :attr:`properties` of an item of unspecified level

        Parameters
        ----------
        item : hashable
            name of an item

        Returns
        -------
        item_key : tuple of (int, hashable)
            ``(level, item)``

        Raises
        ------
        KeyError
            If `item` is not in :attr:`properties`

        Warns
        -----
        UserWarning
            If `item` appears in multiple levels, returns the first (closest to 0)

        """
        try:
            item_loc = self.properties.xs(item, level=1, drop_level=False).index
        except KeyError as ex:  # item not in df
            raise KeyError(f"no properties initialized for 'item': {item}") from ex

        try:
            item_key = item_loc.item()
        except ValueError:
            item_loc, _ = item_loc.sortlevel(sort_remaining=False)
            item_key = item_loc[0]
            warnings.warn(f"item found in multiple levels: {tuple(item_loc)}")
        return item_key

    def set_property(
        self,
        item: T,
        prop_name: Any,
        prop_val: Any,
        level: Optional[int] = None,
    ) -> None:
        """Set a property of an item

        Parameters
        ----------
        item : hashable
            name of an item
        prop_name : hashable
            name of the property to set
        prop_val : any
            value of the property to set
        level : int, optional
            level index of the item;
            required if `item` is not already in :attr:`properties`

        Raises
        ------
        ValueError
            If `level` is not provided and `item` is not in :attr:`properties`

        Warns
        -----
        UserWarning
            If `level` is not provided and `item` appears in multiple levels,
            assumes the first (closest to 0)

        See Also
        --------
        get_property, get_properties
        """
        if level is not None:
            item_key = (level, item)
        else:
            try:
                item_key = self._property_loc(item)
            except KeyError as ex:
                raise ValueError(
                    "cannot infer 'level' when initializing 'item' properties"
                ) from ex

        if prop_name in self.properties:
            self._properties.loc[item_key, prop_name] = prop_val
        else:
            try:
                self._properties.loc[item_key, self._misc_props_col].update(
                    {prop_name: prop_val}
                )
            except KeyError:
                self._properties.loc[item_key, :] = {
                    self._misc_props_col: {prop_name: prop_val}
                }

    def get_property(self, item: T, prop_name: Any, level: Optional[int] = None) -> Any:
        """Get a property of an item

        Parameters
        ----------
        item : hashable
            name of an item
        prop_name : hashable
            name of the property to get
        level : int, optional
            level index of the item

        Returns
        -------
        prop_val : any
            value of the property

        Raises
        ------
        KeyError
            if (`level`, `item`) is not in :attr:`properties`,
            or if `level` is not provided and `item` is not in :attr:`properties`

        Warns
        -----
        UserWarning
            If `level` is not provided and `item` appears in multiple levels,
            assumes the first (closest to 0)

        See Also
        --------
        get_properties, set_property
        """
        if level is not None:
            item_key = (level, item)
        else:
            try:
                item_key = self._property_loc(item)
            except KeyError:
                raise  # item not in properties

        try:
            prop_val = self.properties.loc[item_key, prop_name]
        except KeyError as ex:
            if ex.args[0] == prop_name:
                prop_val = self.properties.loc[item_key, self._misc_props_col].get(
                    prop_name
                )
            else:
                raise KeyError(
                    f"no properties initialized for ('level','item'): {item_key}"
                ) from ex

        return prop_val

    def get_properties(self, item: T, level: Optional[int] = None) -> dict[Any, Any]:
        """Get all properties of an item

        Parameters
        ----------
        item : hashable
            name of an item
        level : int, optional
            level index of the item

        Returns
        -------
        prop_vals : dict
            ``{named property: property value, ...,
            misc. property column name: {property name: property value}}``

        Raises
        ------
        KeyError
            if (`level`, `item`) is not in :attr:`properties`,
            or if `level` is not provided and `item` is not in :attr:`properties`

        Warns
        -----
        UserWarning
            If `level` is not provided and `item` appears in multiple levels,
            assumes the first (closest to 0)

        See Also
        --------
        get_property, set_property
        """
        if level is not None:
            item_key = (level, item)
        else:
            try:
                item_key = self._property_loc(item)
            except KeyError:
                raise

        try:
            prop_vals = self.properties.loc[item_key].to_dict()
        except KeyError as ex:
            raise KeyError(
                f"no properties initialized for ('level','item'): {item_key}"
            ) from ex

        return prop_vals
