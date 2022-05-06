import warnings
from hypernetx import *
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from collections import defaultdict, OrderedDict, UserList
from collections.abc import Hashable
from scipy.sparse import csr_matrix
from hypernetx.classes.helpers import *


class StaticEntity(object):
    """
    A new Entity object using pandas.DataFrame as the base data structure

    TODO: allow addition of rows of data from dict of lists or lists of lists
    TODO: allow removal of rows of data

    Parameters
    ----------
    entity: pandas.DataFrame, dict of lists or sets, list of lists or sets
        If a pandas.DataFrame with N columns, represents N-dimensonal entity
        data, Otherwise, a dict or list represents 2-dimensional entity data
    data : 2d numpy.ndarray
        A sparse representation of an N-dimensional incidence tensor with M
        nonzero cells as an M x N matrix of tensor indices.
    static : bool, default=True
        If True, data may not be altered, and the state dict will never be
        cleared. If False, rows may be added to and removed from data, and
        updates will clear the state dict
    labels : OrderedDict of lists, optional
        User defined labels corresponding to integers in data, ignored if
        data=None
    uid : Hashable, optional
    weights : array-like or Hashable, optional
        User specified cell weights corresponding to data,
        If array-like, length must equal number of rows in data.
        If Hashable, must be the name of a column in data.
        Otherwise, weight for all rows is assumed to be 1.
    aggregateby : str, optional, default='sum'
        Method to aggregate cell weights of duplicate rows of data.
        If None, duplicate rows will be dropped without aggregating cell
        weights

    Attributes
    ----------
    uid: Hashable
    static: bool
    dataframe: pandas.DataFrame
    data_cols: list of Hashables
    cell_weight_col: Hashable
    dimsize: int
    state_dict: dict
        The state_dict holds all attributes that must be recomputed when the
        data is updated. The values for these attributes will be computed as
        needed if they do not already exist in the state_dict.

        data : numpy.ndarray
            sparse tensor indices for incidence tensor
        labels: dict of lists
            labels corresponding to integers in sparse tensor indices given by
            data
        cell_weights: dict
            keys are rows of labeled data as tuples, values are cell weight of
            the row
        dimensions: tuple of ints
            tuple of number of unique labels in each column/dimension of the
            data
        uidset: dict of sets
            keys are columns of the dataframe, values are the set of unique
            labeled values in the column
        elements: defaultdict of nested dicts
            top level keys are column names, elements[col1][col2] holds the
            elements by level dict with col1 as level 1 and col2 as level 2
            (keys are unique values of col1, values are list of unique values
            of col2 that appear in a row of data with the col1 value key)

    """

    def __init__(
        self,
        entity=None,
        data=None,
        static=True,
        labels=None,
        uid=None,
        weights=None,
        aggregateby="sum",
        properties=None
    ):
        # set unique identifier
        self._uid = uid
        self._properties = self._create_properties(properties)

        # if static, the original data cannot be altered
        # the state dict stores all computed values that may need to be updated
        # if the data is altered - the dict will be cleared when data is added
        # or removed
        self._static = static
        self._state_dict = {}

        # entity data is stored in a DataFrame for basic access without the n
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
            if isinstance(labels, dict) and len(labels) == len(self._dataframe
                                                               .columns):
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
        """
        Data array or tensor array of Static Entity

        Returns
        -------
        numpy.ndarray
            Two dimensional array. Each row has system ids of objects in the static entity.
            Each column corresponds to one level of the static entity.

        """
        if "data" not in self._state_dict:
            if self.empty:
                self._state_dict["data"] = np.zeros((0, 0), dtype=int)
            else:
                # assumes dtype of data cols is categorical and dataframe not
                # altered
                self._state_dict["data"] = (
                    self._dataframe[self._data_cols]
                    .apply(lambda x: x.cat.codes)
                    .to_numpy()
                )

        return self._state_dict["data"]

    @property
    def labels(self):
        """
        Dictionary of labels

        Returns
        -------
        dict
            User defined identifiers for objects in static entity. Ordered keys correspond
            levels. Ordered values correspond to integer representation of values in data.
        """
        if "labels" not in self._state_dict:
            # assumes dtype of data cols is categorical and dataframe not
            # altered
            self._state_dict["labels"] = {
                col: self._dataframe[col].cat.categories.to_list()
                for col in self._data_cols
            }

        return self._state_dict["labels"]

    @property
    def cell_weights(self):
        """
        User defined weights corresponding to unique rows in data.

        Returns
        -------
        dict
            Dictionary of values aligned to data.
        """
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
        """
        Dimension of data

        Returns
        -------
        tuple
            Tuple of number of distinct labels in each level, ordered by level.
        """
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
        """
        Number of categories in the data

        Returns
        -------
        int
            Number of levels in static entity, equals length of self.dimensions
        """
        return self._dimsize

    @property
    def properties(self):
        #Dev Note: Not sure what this contains, when running tests it contained an empty pandas series
        """
        Pandas series containing properties for the Static Entity.

        Returns
        -------
        pandas.Series
            Properties for the Static Entity
        """

        return self._properties

    @property
    def uid(self):
        # Dev Note: This also returned nothing in my harry potter dataset, not sure if it was supposed to contain anything
        """
        User defined identifier for each object in static entity.

        Returns
        -------
        str, int
            Identifiers, which distinguish  objects within each level.
        """
        return self._uid

    @property
    def uidset(self):
        """
        Returns a set of the string identifiers for Static Entity

        Returns
        -------
        frozenset
            Hashable set of keys.
        """
        return self.uidset_by_level(0)

    @property
    def children(self):
        """
        Labels of keys of first index

        Returns
        -------
        frozenset
            Set of labels in the second level.
        """
        return self.uidset_by_level(1)

    def uidset_by_level(self, level):
        """
        The labels found in a specific level

        Parameters
        ----------
        level : int

        Returns
        -------
        frozenset
        """
        col = self._data_cols[level]
        return self.uidset_by_column(col)

    def uidset_by_column(self, column):
        # Dev Note: This threw an error when trying it on the harry potter dataset,
        # when trying 0, or 1 for column. I'm not sure how this should be used
        """
        The labels found in a specific column of the data

        Parameters
        ----------
        level : int

        Returns
        -------
        frozenset
        """
        if "uidset" not in self._state_dict:
            self._state_dict["uidset"] = {}
        if column not in self._state_dict["uidset"]:
            self._state_dict["uidset"][column] = set(
                self._dataframe[column].dropna().unique()
            )

        return self._state_dict["uidset"][column]

    @property
    def elements(self):
        """
        Keys and values in the order of insertion

        Returns
        -------
        dict
            Same as elements_by_level with level1 = 0, level2 = 1.
            Compare with EntitySet with level1 = elements, level2 = children.

        """
        if self._dimsize == 1:
            return {k: AttrList(entity=self, key=(0, k)) for k in self.uidset}

        return self.elements_by_level(0, 1)

    @property
    def incidence_dict(self):
        """
        Same as elements.

        Returns
        -------
        dict
            Same as elements_by_level with level1 = 0, level2 = 1.
            Compare with EntitySet with level1 = elements, level2 = children.
        """
        return self.elements

    @property
    def memberships(self):
        """
        Reverses the elements dictionary

        Returns
        -------
        dict
            Same as elements_by_level with level1 = 1, level2 = 0.
        """
        return self.elements_by_level(1, 0)

    def elements_by_level(self, level1, level2):
        """
        Elements of Static Entity by specified level(s)

        Parameters
        ----------
        level1 : int, optional
            edges
        level2 : int, optional
            nodes

        Returns
        -------
        dict

        think: level1 = edges, level2 = nodes
        """
        col1 = self._data_cols[level1]
        col2 = self._data_cols[level2]
        return self.elements_by_column(col1, col2)

    def elements_by_column(self, col1, col2):
        # Dev Note: This threw an error when trying it on the harry potter dataset,
        # when trying 0, or 1 for column. I'm not sure how this should be used
        """
        Elements of Static Entity by specified column(s)

        Parameters
        ----------
        level1 : int, optional
            edges
        level2 : int, optional
            nodes

        Returns
        -------
        dict

        think: level1 = edges, level2 = nodes
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
        """
        Pandas dataframe representation of the data

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._dataframe

    @property
    def isstatic(self):
        #Dev Note: I'm guessing this is no longer necessary?
        return self._static

    def size(self, level=0):
        """
        The number of elements in E, the size of dimension 0 in the E.arr

        Returns
        -------
        int
        """
        return self.dimensions[level]

    @property
    def empty(self):
        """
        Checks if the dimsize = 0

        Returns
        -------
        bool
        """
        return self._dimsize == 0

    def is_empty(self, level=0):
        """
        Checks if a level is empty

        Returns
        -------
        bool
        """
        return self.empty or self.size(level) == 0

    def __len__(self):
        """
        Returns the number of elements in Static Entity

        Returns
        -------
        int
            Number of distinct labels in level 0.
        """
        return self.dimensions[0]

    def __contains__(self, item):
        """
        Defines containment for StaticEntity based on labels/categories.

        Parameters
        ----------
        item : string

        Returns
        -------
        bool
        """
        for labels in self.labels.values():
            if item in labels:
                return True
        return False

    def __getitem__(self, item):
        """
        Get value of key in E.elements

        Parameters
        ----------
        item : string

        Returns
        -------
        list
        """
        return self.elements[item]

    def __iter__(self):
        """
        Create iterator from E.elements

        Returns
        -------
        odict_iterator
        """
        return iter(self.elements)

    def __call__(self, label_index=0):
        """
        Allows user to create instance of class that behaves like a function
        """
        return iter(self.labels[self._data_cols[label_index]])

    def __repr__(self):
        """
        Returns a string resembling the constructor for staticentity without
        any children

        Returns
        -------
        string
        """
        return (
            self.__class__.__name__ + f"({self._uid}, {list(self.uidset)},
                                         {[] if self.properties.empty
                                         else self.properties.droplevel(0)
                                         .to_dict()})"
        )

    def index(self, column, value=None):
        """
        Returns dimension of category and index of value

        Parameters
        ----------
        category : string
        value : string, optional

        Returns
        -------
        int or tuple of ints
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
        """
        Returns dimension of category and index of values (array)

        Parameters
        ----------
        category : string
        values : single string or array of strings

        Returns
        -------
        list
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
        """
        Replaces a category index and value index with label

        Parameters
        ----------
        level : int
            category index of label
        index : int
            value index of label

        Returns
        -------
         str
        """
        column = self._data_cols[level]

        if isinstance(index, (int, np.integer)):
            return self.labels[column][index]

        return [self.labels[column][i] for i in index]

    def translate_arr(self, coords):
        """
        Translates a single cell in the entity array

        Parameters
        ----------
        coords : tuple of ints

        Returns
        -------
        list
        """
        assert len(coords) == self._dimsize
        translation = []
        for level, index in enumerate(coords):
            translation.append(self.translate(level, index))

        return translation

    def level(self, item, min_level=0, max_level=None, return_index=True):
        """
        Returns first level item appears by order of keys from minlevel to
        maxlevel inclusive

        Parameters
        ----------
        item : string
        min_level : int, optional
        max_level : int, optional

        return_index : bool, optional

        Returns
        -------
        str
        """
        if max_level is None or max_level >= self._dimsize:
            max_level = self._dimsize - 1

        columns = self._data_cols[min_level: max_level + 1]
        levels = range(min_level, max_level + 1)

        for col, lev in zip(columns, levels):
            if item in self.labels[col]:
                if return_index:
                    return self.index(col, item)

                return lev

        print(f'"{item}" not found.')
        return None

    def add(self, *args):
        """
        Adds unpacked arguments(args) to entity elements. Depends on
        add_element()

        Parameters
        ----------
        args : One or more data representations of an Entity

        Returns
        -------
        self : StaticEntity

        Note
        ----
        Adding an element to an object in a hypergraph will not add the
        element to the hypergraph and will cause an error. Use
        :func:`Hypergraph.add_edge <classes.hypergraph.Hypergraph.add_edge>` or
        :func:`Hypergraph.add_node_to_edge <classes.hypergraph.Hypergraph \
            .add_node_to_edge>` instead.

        """
        for item in args:
            self.add_element(item)
        return self

    def add_elements_from(self, arg_set):
        """
        Similar to :func:`add()` it allows for adding from an interable.

        Parameters
        ----------
        arg_set : Iterable of data representations of an StaticEntity

        Returns
        -------
        self : StaticEntity

        """
        for item in arg_set:
            self.add_element(item)
        return self

    def add_element(self, data):
        """
        Converts the data to a dataframe then utilizes
        :func`_add_from_dataframe()` to append the dataframe of new data to
        the existing underlying dataframe representation of the StaticEntity.

        Parameters
        ----------
        data : Data representation of Hypergraph edges and/or nodes.

        Returns
        -------
        self : StaticEntity

        """
        if isinstance(data, StaticEntity):
            df = data.dataframe
            self.__add_from_dataframe(df)

        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
            self.__add_from_dataframe(df)

        if isinstance(data, pd.DataFrame):
            self.__add_from_dataframe(data)

        return self

    def __add_from_dataframe(self, df):
        """
        Takes a new dataframe of nodes and edges and appends to the existing
        dataframe representation of the StaticEntity.
        Parameters
        ----------
        data : Data representation of Hypergraph edges and/or nodes.

        Returns
        -------
        self : StaticEntity

        """
        if all(col in df for col in self._data_cols):
            new_data = pd.concat((self._dataframe, df), ignore_index=True)
            new_data[self._cell_weight_col] = new_data[self._cell_weight_col] \
                .fillna(1)

            self._dataframe, _ = remove_row_duplicates(
                new_data,
                self._data_cols,
                weights=self._cell_weight_col,
            )

            self._dataframe[self._data_cols] = self. \
                _dataframe[self._data_cols].astype("category")

            self._state_dict.clear()

    def remove(self, *args):
        """
        Removes nodes or edges from a StaticEntity if they exist in the
        StaticEntity

        Parameters
        ----------
        args : One or more data representations of a StaticEntities elements

        Returns
        -------
        self : StaticEntity


        """
        for item in args:
            self.remove_element(item)
        return self

    def remove_elements_from(self, arg_set):
        """
        Similar to :func:`remove()`. Removes elements in arg_set.

        Parameters
        ----------
        arg_set : Dev Note: Ask Brenda about phrasing

        Returns
        -------
        self : StaticEntity

        """
        for item in arg_set:
            self.remove_element(item)
        return self

    def remove_element(self, item):
        """
        Removes item from dataframe.

        Parameters
        ----------
        item : Data representation of StaticEntity

        Returns
        -------
        self : StaticEntity

        """
        updated_dataframe = self._dataframe

        for column in self._dataframe:
            updated_dataframe = updated_dataframe[updated_dataframe[column]
                                                  != item]

        self._dataframe, _ = remove_row_duplicates(
            updated_dataframe,
            self._data_cols,
            weights=self._cell_weight_col,
        )
        self._dataframe[self._data_cols] = self._dataframe[self._data_cols] \
            .astype("category")

        self._state_dict.clear()
        for col in self._data_cols:
            self._dataframe[col] = self._dataframe[col].cat \
                .remove_unused_categories()

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
        """
        Convenience method to navigate large tensor

        Parameters
        ----------
        level1 : int, optional
            indexes columns
        level2 : int, optional
            indexes rows
        weights : bool, dict optional, default=False
            If False all nonzero entries are 1.
            If True all nonzero entries are filled by self.cell_weight
            dictionary values, use :code:`aggregateby` to specify how duplicate
            entries should have weights aggregated.
            If dict, keys must be in (edge.uid, node.uid) form; only nonzero
            cells in the incidence matrix will be updated by dictionary.
        aggregateby : str, optional, {None, 'last', count', 'sum', 'mean',
            'median', max', 'min', 'first', 'last'}, default : 'count'
            Method to aggregate weights of duplicate rows in data. If None,
            then all cell weights will be set to 1.
        index : bool, optional

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse matrix representation of incidence matrix for two levels of
            static entity.

        Note
        ----
        In the context of hypergraphs think level1 = edges, level2 = nodes
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

    def restrict_to_levels(self, levels, weights=False, aggregateby="sum",
                           **kwargs):
        """
        Limit Static Entity data to specific levels

        Parameters
        ----------
        levels : array
            index of labels in data
        weights : bool, optional, default : False
            Whether or not to aggregate existing weights in self when
            restricting to levels. If False then weights will be assigned 1.
        aggregateby : str, optional, {None, 'last', count', 'sum', 'mean',
            'median', max', 'min', 'first', 'last'}, default : 'count' Method
            to aggregate cell_weights of duplicate rows in setsystem of type
            pandas.DataFrame. If None then all cell_weights will be set to 1.
        uid : None, optional

        Returns
        -------
        Static Entity class
            hnx.classes.staticentity.StaticEntity
        """
        levels = [lev for lev in levels if lev < self._dimsize]

        if levels:
            cols = [self._data_cols[lev] for lev in levels]

            weights = self._cell_weight_col if weights else None

            if weights:
                cols.append(weights)

            entity = self._dataframe[cols]

            kwargs.update(entity=entity, weights=weights,
                          aggregateby=aggregateby)

        return self.__class__(**kwargs)

    def restrict_to_indices(self, indices, level=0, **kwargs):
        """
        Limit Static Entity data to specific indices of keys

        Parameters
        ----------
        indices : array
            array of category indices
        level : int, optional
            index of label
        uid : None, optional

        Returns
        -------
        Static Entity class
            hnx.classes.staticentity.StaticEntity
        """
        column = self._dataframe[self._data_cols[level]]
        values = column.cat.categories[list(indices)]
        entity = self._dataframe.loc[column.isin(values)]
        for col in self._data_cols:
            entity.loc[:, col] = entity[col].cat.remove_unused_categories()
        return self.__class__(entity=entity, **kwargs)

    def _create_properties(self, props):
        """
        Create new properties

        Parameters
        ----------
        props : list of properties

        Returns
        -------
        pandas.Series
        """
        index = pd.MultiIndex(levels=([], []), codes=([], []), names=('level',
                                                                      'item'))
        kwargs = {'index': index, 'name': 'properties'}
        if props:
            levels = [self.level(item, return_index=False) for item in props]
            levels_items = [(lev, item) for lev, item in zip(levels, props) if
                            lev is not None]
            index = pd.MultiIndex.from_tuples(levels_items, names=('level',
                                                                   'item'))
            data = [props[item] for _, item in index]
            kwargs.update(index=index, data=data)
        return pd.Series(**kwargs)

    def assign_properties(self, props):
        #Dev Note: Not sure what the put here.
        properties = self._create_properties(props)

        if not self._properties.empty:
            properties = update_properties(self._properties, properties)

        self._properties = properties
