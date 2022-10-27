from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Hashable, Mapping, Sequence, Iterable
from typing import Union, TypeVar, Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hypernetx.classes.helpers import (
    AttrList,
    assign_weights,
    create_properties,
    remove_row_duplicates,
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
        ``[item level, item label, optional named properties,
        {property name: property value}]``
        (order of columns does not matter; see note for an example).
        If doubly-nested dict,
        ``{item level: {item label: {property name: property value}}}``.
    props_col, level_col, id_col : str, default="properties", "level, "id"
        Column names for miscellaneous properties, level index, and item name in
        :attr:`properties`; see Notes for explanation.

    Notes
    -----
    A property is a named attribute assigned to a single item in the data.

    You can pass a **table of properties** to `properties` as a ``DataFrame``:

    +-------+---------+--------------------------+-------+------------------+
    | Level | ID      | [explicit property type] | [...] | misc. properties |
    +=======+=========+==========================+=======+==================+
    | 0     | level 0 | property value           | ...   | {property name:  |
    |       | item    |                          |       | property value}  |
    +-------+---------+--------------------------+-------+------------------+
    | 1     | level 1 | property value           | ...   | {property name:  |
    |       | item    |                          |       | property value}  |
    +-------+---------+--------------------------+-------+------------------+
    | ...   | ...     | ...                      | ...   | ...              |
    +-------+---------+--------------------------+-------+------------------+
    | N     | level N | property value           | ...   | {property name:  |
    |       | item    |                          |       | property value}  |
    +-------+---------+--------------------------+-------+------------------+

    The names of the Level and ID columns must be specified by `level_col` and `id_col`.
    `props_col` can be used to specify the nme of the column to be used for
    miscellaneous properties; if no column by that name is found, a new column will be
    created and populated with empty ``dicts``. All other columns will be considered
    explicit property types. The order of the columns does not matter.

    This method assumes that there are no row duplicates in the `properties` table;
    if duplicates are found, all but the first occurrence will be dropped.

    """

    def __init__(
        self,
        entity: Optional[
            pd.DataFrame | Mapping[T, Iterable[T]] | Iterable[Iterable[T]]
        ] = None,
        data: Optional[np.ndarray] = None,
        static: bool = False,
        labels: Optional[OrderedDict[T, Sequence[T]]] = None,
        uid: Optional[Hashable] = None,
        weights: Optional[Sequence[float] | str] = None,
        aggregateby: Optional[str] = "sum",
        properties: Optional[pd.DataFrame | dict[int, dict[T, dict[Any, Any]]]] = None,
        props_col: str = "properties",
        level_col: str = "level",
        id_col: str = "id",
    ):

        # set unique identifier
        self._uid = uid

        # create properties
        self._props_col = props_col
        self._properties = create_properties(properties, [level_col, id_col], props_col)

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
            self._dataframe = pd.DataFrame({0: entity.index, 1: entity.values})

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
        # TODO: clean this up to be less hacky?
        else:
            self._dataframe = pd.DataFrame()
            self._data_cols = []
            self._cell_weight_col = None
            self._dimsize = 0
            return

        # assign a new or existing column of the dataframe to hold cell weights
        self._dataframe, self._cell_weight_col = assign_weights(
            self._dataframe, weights=weights
        )
        # store a list of columns that hold entity data (not properties or
        # weights)
        self._data_cols = list(self._dataframe.columns.drop(self._cell_weight_col))

        # each entity data column represents one dimension of the data
        # (data updates can only add or remove rows, so this isn't stored in
        # state dict)
        self._dimsize = len(self._data_cols)

        # remove duplicate rows and aggregate cell weights as needed
        self._dataframe, _ = remove_row_duplicates(
            self._dataframe,
            self._data_cols,
            weights=self._cell_weight_col,
            aggregateby=aggregateby,
        )

        # set the dtype of entity data columns to categorical (simplifies
        # encoding, etc.)
        self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
            "category"
        )

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
    def properties(self):
        # Dev Note: Not sure what this contains, when running tests it contained an empty pandas series
        """Properties assigned to items in the underlying data table

        Returns
        -------
        pandas.Series
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
        memberships :
            dual of this representation,
            i.e., each item in level 1 (second column) defines a set
        elements_by_level, elements_by_column :
            system of sets representation of any two levels (columns); specified by level index or column name

        """
        if self._dimsize == 1:
            return {k: AttrList(entity=self, key=(0, k)) for k in self.uidset}

        return self.elements_by_level(0, 1)

    @property
    def incidence_dict(self):
        """System of sets representation of the first two levels (columns) of the underlying data table

        .. deprecated:: 2.0.0
            Duplicates `elements`

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {level 0 item : AttrList(level 1 items)}

        """
        return self.elements

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
        # Dev Note: This threw an error when trying it on the harry potter dataset,
        # when trying 0, or 1 for column. I'm not sure how this should be used
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
            elements = self._dataframe.groupby(col1)[col2].unique().to_dict()
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

    def __repr__(self):
        """String representation of the Entity

        e.g., "Entity(uid, [level 0 items], {item: {property name: property value}})"

        Returns
        -------
        str
        """
        return (
            self.__class__.__name__
            + f"""({self._uid}, {list(self.uidset)},
                                         {[] if self.properties.empty
        else self.properties.droplevel(0)
                .to_dict()})"""
        )

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
    ):
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
            sparse representation of incidence matrix

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

    def restrict_to_levels(self, levels, weights=False, aggregateby="sum", **kwargs):
        """Create a new Entity by restricting the underlying data table to a subset of levels (columns)

        Parameters
        ----------
        levels : iterable of int
            indices of a subset of levels (columns) of data
        weights : bool, default=False
            If True, aggregate existing cell weights to get new cell weights
            Otherwise, all new cell weights will be 1
        aggregateby : {'last', count', 'sum', 'mean','median', max', 'min', 'first', 'last', None}, default='count'
            Method to aggregate weights of duplicate rows in data table
            If None or weights=False then all new cell weights will be 1
        **kwargs
            Extra arguments to `Entity` constructor

        Returns
        -------
        Entity

        See Also
        --------
        EntitySet
        """
        levels = [lev for lev in levels if lev < self._dimsize]

        if levels:
            cols = [self._data_cols[lev] for lev in levels]

            weights = self._cell_weight_col if weights else None

            if weights:
                cols.append(weights)

            entity = self._dataframe[cols]

            if not self.properties.empty:
                new_levels = {old: new for new, old in enumerate(levels)}
                properties = self.properties.loc[levels].reset_index()
                properties.level = properties.level.map(new_levels)

                kwargs.update(properties=properties)

            kwargs.update(
                entity=entity,
                weights=weights,
                aggregateby=aggregateby,
                props_col=self._props_col,
            )

        return self.__class__(**kwargs)

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
        restricted = self.__class__(entity=entity, props_col=self._props_col, **kwargs)

        if not self.properties.empty:
            prop_idx = [
                (lv, uid)
                for lv in range(restricted.dimsize)
                for uid in restricted.uidset_by_level(lv)
            ]
            properties = self.properties.loc[prop_idx].reset_index()
            restricted.assign_properties(properties)
        return restricted

    def assign_properties(
        self,
        props: pd.DataFrame | dict[int, dict[T, dict[Any, Any]]],
        level_col: Optional[str] = None,
        id_col: Optional[str] = None,
        misc_col: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Assign new properties to items in the data table and update `self.properties`

        Parameters
        ----------
        props : pandas.DataFrame, dict of iterables, doubly-nested dict, or None
            See documentation of the `properties` parameter in :class:`Entity`
        level_col, id_col, misc_col : str, optional
            column names corresponding to the levels, items, and misc. properties;
            if None, default to :attr:`_level_col`, :attr:`_id_col`, :attr:`_props_col`,
            respectively.
        replace: bool, default=False
            If True, replace existing :attr:`properties` with result;
            otherwise update with new values from result


        See Also
        --------
        properties, update_properties
        """
        level_col = level_col or self.properties.index.names[0]
        id_col = id_col or self.properties.index.names[1]
        misc_col = misc_col or self._props_col
        # convert properties to MultiIndexed DataFrame
        properties = create_properties(props, [level_col, id_col], misc_col)
        if level_col != self.properties.index.names[0]:
            properties.index.set_names(
                self.properties.index.names[0], level=0, inplace=True
            )
        if id_col != self.properties.index.names[1]:
            properties.index.set_names(
                self.properties.index.names[1], level=1, inplace=True
            )
        if misc_col != self._props_col:
            properties.rename(columns={misc_col: self._props_col}, inplace=True)

        if replace:
            self._properties = properties
        else:
            self._properties = self._properties.combine_first(properties)
            self._properties.update(properties)

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
        except KeyError:  # item not in df
            raise KeyError(f"no properties initialized for 'item': {item}")

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
            self._properties.loc[item_key, prop_name] = pd.Series([prop_val])
        else:
            self._properties.loc[item_key, self._props_col].update(
                {prop_name: prop_val}
            )

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
                prop_val = self.properties.loc[item_key, self._props_col].get(prop_name)
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
