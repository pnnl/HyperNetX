from collections import OrderedDict, defaultdict, UserList
from collections.abc import Iterable
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import *
from hypernetx.exception import HyperNetXError
from hypernetx.classes.entity import Entity, EntitySet
from hypernetx.utils import (
    HNXCount,
    DefaultOrderedDict,
    remove_row_duplicates,
    reverse_dictionary,
)
from scipy.sparse import coo_matrix, csr_matrix, issparse
import itertools as it
import pandas as pd

__all__ = ["StaticEntity", "StaticEntitySet"]


class StaticEntity(object):

    """
    .. _staticentity:

    Parameters
    ----------
    entity : StaticEntity, StaticEntitySet, Entity, EntitySet, pandas.DataFrame, dict, or list of lists
        If a pandas.DataFrame, an error will be raised if there are nans.
    data : array or array-like
        Two dimensional array of integers. Provides sparse tensor indices for incidence
        tensor.
    arr : numpy.ndarray or scip.sparse.matrix, optional, default=None
        Incidence tensor of data.
    labels : OrderedDict of lists, optional, default=None
        User defined labels corresponding to integers in data.
    uid : hashable, optional, default=None
    weights : array-like, optional, default : None
        User specified weights corresponding to data, length must equal number
        of rows in data. If None, weight for all rows is assumed to be 1.
    keep_weights : bool, optional, default : True
        Whether or not to use existing weights when input is StaticEntity, or StaticEntitySet.
    aggregateby : str, optional, {'count', 'sum', 'mean', 'median', max', 'min', 'first', 'last', None}, default : 'count'
        Method to aggregate cell_weights of duplicate rows if setsystem  is of type pandas.DataFrame of
        StaticEntity. If None all cell weights will be set to 1.

    props : user defined keyword arguments to be added to a properties dictionary, optional

    Attributes
    ----------
    properties : dict
        Description

    """

    def __init__(
        self,
        entity=None,
        data=None,
        arr=None,
        labels=None,
        uid=None,
        weights=None, ### in this context weights is just a column of values corresponding to the rows in data.
        keep_weights=True,
        aggregateby="sum",
        **props,
    ):
        self._uid = uid
        self.properties = {}
        if entity is not None:
            if isinstance(entity, StaticEntity) or isinstance(entity, StaticEntitySet):
                self.properties.update(entity.properties)
                self.properties.update(props)
                self.__dict__.update(self.properties)
                self.__dict__.update(props)
                self._data = entity._data.copy()
                if keep_weights:
                    self._weights = entity._weights
                    self._cell_weights = dict(entity._cell_weights)
                else:
                    self._data, self._cell_weights = remove_row_duplicates(
                        entity.data, weights=weights, aggregateby=aggregateby
                    )
                self._dimensions = entity._dimensions
                self._dimsize = entity._dimsize
                self._labels = OrderedDict(
                    (category, np.array(values))
                    for category, values in entity._labels.items()
                )
                self._keys = np.array(list(self._labels.keys()))
                # returns the index of the category (column)
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }
                self._arr = None
            elif isinstance(entity, pd.DataFrame):
                self.properties.update(props)
                (
                    self._data,
                    self._labels,
                    self._cell_weights,
                ) = _turn_dataframe_into_entity(
                    entity, weights=weights, aggregateby=aggregateby
                )
                self.__dict__.update(self.properties)
                self._arr = None
                self._dimensions = tuple([max(x) + 1 for x in self._data.transpose()])
                self._dimsize = len(self._dimensions)
                self._keys = np.array(list(self._labels.keys()))
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }

            else:  # For these cases we cannot yet add cell_weights directly, cell_weights default to duplicate counts
                if isinstance(entity, Entity) or isinstance(entity, EntitySet):
                    d = entity.incidence_dict
                    (
                        self._data,
                        self._labels,
                        self._cell_weights,
                    ) = _turn_dict_to_staticentity(
                        d
                    )  # For now duplicate entries will be removed.
                elif isinstance(entity, dict):  # returns only 2 levels
                    (
                        self._data,
                        self._labels,
                        self._cell_weights,
                    ) = _turn_dict_to_staticentity(
                        entity
                    )  # For now duplicate entries will be removed.
                else:  # returns only 2 levels
                    (
                        self._data,
                        self._labels,
                        self._cell_weights,
                    ) = _turn_iterable_to_staticentity(entity)
                self._dimensions = tuple([len(self._labels[k]) for k in self._labels])
                self._dimsize = len(self._dimensions)  # number of columns
                self._keys = np.array(
                    list(self._labels.keys())
                )  # These are the column headers from the dataframe
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }
                self.properties.update(props)
                self.__dict__.update(
                    self.properties
                )  # Add function to set attributes ###########!!!!!!!!!!!!!
                self._arr = None
        elif data is not None:
            self._arr = None
            self._data, self._cell_weights = remove_row_duplicates(
                data, weights=weights, aggregateby=aggregateby
            )
            self._dimensions = tuple([max(x) + 1 for x in self._data.transpose()])
            self._dimsize = len(self._dimensions)
            self.properties.update(props)
            self.__dict__.update(props)
            if labels is not None:
                self._labels = OrderedDict(
                    (category, np.array(values)) for category, values in labels.items()
                )  # OrderedDict(category,np.array([categorical values ....])) is aligned to arr
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._keys = np.array(list(labels.keys()))
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }
            else:
                self._labels = OrderedDict(
                    [
                        (int(dim), np.arange(ct))
                        for dim, ct in enumerate(self.dimensions)
                    ]
                )
                self._keyindex = defaultdict(_fd)
                self._keys = np.arange(self._dimsize)
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }

        elif arr is not None:
            self._arr = arr
            self.properties.update(props)
            self.__dict__.update(props)
            self._state_dict = {"arr": arr * 1}
            self._dimensions = arr.shape
            self._dimsize = len(arr.shape)
            self._data, self._cell_weights = _turn_tensor_to_data(arr * 1)
            if labels is not None:
                self._labels = OrderedDict(
                    (category, np.array(values)) for category, values in labels.items()
                )
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._keys = np.array(list(labels.keys()))
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }

            else:
                self._labels = OrderedDict(
                    [
                        (int(dim), np.arange(ct))
                        for dim, ct in enumerate(self.dimensions)
                    ]
                )
                self._keyindex = defaultdict(_fd)
                self._keys = np.arange(self._dimsize)
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }
        else:  # no entity, data or arr is given

            if labels is not None:
                self._labels = OrderedDict(
                    (category, np.array(values)) for category, values in labels.items()
                )
                self._dimensions = tuple([len(labels[k]) for k in labels])
                self._data = np.zeros((0, len(labels)), dtype=int)
                self._cell_weights = {}
                self._arr = np.empty(self._dimensions, dtype=int)
                self._state_dict = {"arr": np.empty(self.dimensions, dtype=int)}
                self._dimsize = len(self._dimensions)
                self._keyindex = dict(
                    zip(self._labels.keys(), np.arange(self._dimsize))
                )
                self._keys = np.array(list(labels.keys()))
                self._index = {
                    cat: dict(zip(self._labels[cat], np.arange(len(self._labels[cat]))))
                    for cat in self._keys
                }
            else:
                self._data = np.zeros((0, 0), dtype=int)
                self._cell_weights = {}
                self._arr = np.array([], dtype=int)
                self._labels = OrderedDict([])
                self._dimensions = tuple([])
                self._dimsize = 0
                self._keyindex = defaultdict(_fd)
                self._keys = np.array([])
                # self._index = lambda category, value: None
                self._index = {
                    cat: dict(
                        zip(self._labels[cat], [None for i in len(self._labels[cat])])
                    )
                    for cat in self._keys
                }

            # if labels is a list of categorical values, then change it into an
            # ordered dictionary?
            self.properties = props
            self.__dict__.update(props)  # keyed by the method name and signature

        if len(self._labels) > 0:
            self._labs = {
                kdx: self._labels.get(self._keys[kdx], {})
                for kdx in range(self._dimsize)
            }
        else:
            self._labs = {}

        self._weights = [self._cell_weights[tuple(t)] for t in self._data]
        self._memberships = None

    @property
    def arr(self):
        """
        Tensor like representation of data indexed by labels with values given by incidence or cell weight.

        Returns
        -------
        numpy.ndarray
            A Numpy ndarray with dimensions equal dimensions of static entity. Entries are cell_weights.
            self.data gives a list of nonzero coordinates aligned with cell_weights.
        """
        if self._arr is not None:
            if type(self._arr) == int and self._arr == 0:
                print("arr cannot be computed")
        else:
            try:
                imat = np.zeros(self.dimensions, dtype=int)
                for d in self._data:
                    imat[tuple(d)] = self._cell_weights[tuple(d)]
                self._arr = imat
            except Exception as ex:
                print(ex)
                print("arr cannot be computed")
                self._arr = 0
        return self._arr  # Do we need to return anything here

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

        return np.array(self._data)

    @property
    def cell_weights(self):
        """
        User defined weights corresponding to unique rows in data.

        Returns
        -------
        numpy.array
            One dimensional array of values aligned to data.
        """
        return dict(self._cell_weights)

    @property
    def labels(self):
        """
        Ordered dictionary of labels

        Returns
        -------
        collections.OrderedDict
            User defined identifiers for objects in static entity. Ordered keys correspond
            levels. Ordered values correspond to integer representation of values in data.
        """
        return dict(self._labels)

    @property
    def dimensions(self):
        """
        Dimension of Static Entity data

        Returns
        -------
        tuple
            Tuple of number of distinct labels in each level, ordered by level.
        """
        return self._dimensions

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
    def keys(self):
        """
        Array of keys of labels

        Returns
        -------
        np.ndarray
            Array of label keys, ordered by level.
        """
        return self._keys

    def keyindex(self, category):
        """
        Returns the index of a category in keys array

        Returns
        -------
        int
            Index osition of particular label in keys equal to the level of the
            category.
        """
        return self._keyindex[category]

    @property
    def uid(self):
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
            Hashable  set of keys.
        """
        return self.uidset_by_level(0)

    @property
    def elements(self):
        """
        Keys and values in the order of insertion

        Returns
        -------
        collections.OrderedDict
            Same as elements_by_level with level1 = 0, level2 = 1.
            Compare with EntitySet with level1 = elements, level2 = children.

        """
        try:
            return dict(self._elements)
        except:
            if len(self._keys) == 1:
                self._elements = {k: UserList() for k in self._labels[self._keys[0]]}
                return dict(self._elements)
            else:
                self._elements = self.elements_by_level(0, translate=True)
                return dict(self._elements)

    @property
    def memberships(self):
        """
        Reverses the elements dictionary

        Returns
        -------
        collections.OrderedDict
            Same as elements_by_level with level1 = 1, level2 = 0.
        """
        try:
            return dict(self._memberships)
        except:
            # self._memberships = reverse_dictionary(self.elements)
            # return self._memberships
            if len(self._keys) == 1:
                return None
            else:
                self._memberships = self.elements_by_level(1, 0, translate=True)
                return dict(self._memberships)

    @property
    def children(self):
        """
        Labels of keys of first index

        Returns
        -------
        numpy.array
            One dimensional array of labels in the second level.

        """
        try:
            return set(self._labs[1])
        except:
            return

    @property
    def incidence_dict(self):
        """
        Same as elements.

        Returns
        -------
        collections.OrderedDict
            Same as elements_by_level with level1 = 0, level2 = 1.
            Compare with EntitySet with level1 = elements, level2 = children.
        """
        return self.elements_by_level(0, translate=True)

    @property
    def dataframe(self):
        """
        Returns the entity data in DataFrame format

        Returns
        -------
        pandas.core.frame.DataFrame
            Dataframe of user defined labels and keys as columns.
        """
        return self.turn_entity_data_into_dataframe(self.data)

    def __len__(self):
        """
        Returns the number of elements in Static Entity

        Returns
        -------
        int
            Number of distinct labels in level 0.
        """
        return self._dimensions[0]

    def __str__(self):
        """
        Return the Static Entity uid

        Returns
        -------
        string
        """
        return f"{self.uid}"

    def __repr__(self):
        """
        Returns a string resembling the constructor for staticentity without any
        children

        Returns
        -------
        string
        """
        return f"StaticEntity({self._uid},{list(self.uidset)},{self.properties})"

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
        return item in np.concatenate(list(self._labels.values()))

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
        # return self.elements_by_level(0, 1, translate=True)[item]
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
        return iter(self._labs[label_index])

    def size(self):
        """
        The number of elements in E, the size of dimension 0 in the E.arr

        Returns
        -------
        int
        """
        return len(self)

    def labs(self, kdx):
        """
        Retrieve labels by index in keys

        Parameters
        ----------
        kdx : int
            index of key in E.keys

        Returns
        -------
        np.ndarray
        """
        return self._labs[kdx]

    def is_empty(self, level=0):
        """
        Boolean indicating if entity.elements is empty

        Parameters
        ----------
        level : int, optional

        Returns
        -------
        bool
        """
        return len(self._labs[level]) == 0

    def uidset_by_level(self, level=0):
        """
        The labels found in columns = level

        Parameters
        ----------
        level : int, optional

        Returns
        -------
        frozenset
        """
        return frozenset(self._labs[level])  # should be update this to tuples?

    def elements_by_level(self, level1=0, level2=None, translate=False):
        """
        Elements of staticentity by specified column

        Parameters
        ----------
        level1 : int, optional
            edges
        level2 : int, optional
            nodes
        translate : bool, optional
            whether to replace indices with labels

        Returns
        -------
        collections.defaultdict

        think: level1 = edges, level2 = nodes
        """
        # Is there a better way to view a slice of self._arr?
        if level1 > self.dimsize - 1 or level1 < 0:
            print(f"This StaticEntity has no level {level1}.")
            return
        if level2 is None:
            level2 = level1 + 1

        if level2 > self.dimsize - 1 or level2 < 0:
            print(f"This StaticEntity has no level {level2}.")
            return
            # elts = OrderedDict([[k, UserList()] for k in self._labs[level1]])
        elif level1 == level2:
            print(f"level1 equals level2")
            return
            # elts = OrderedDict([[k, UserList()] for k in self._labs[level1]])

        temp, _ = remove_row_duplicates(self.data[:, [level1, level2]])
        elts = DefaultOrderedDict(UserList)
        for row in temp:
            elts[row[0]].append(row[1])

        if translate:
            telts = DefaultOrderedDict(UserList)
            for kdx, vec in elts.items():
                k = self._labs[level1][kdx]
                for vdx in vec:
                    telts[k].append(self._labs[level2][vdx])
            return telts
        else:
            return elts

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
            If dict, keys must be in (edge.uid, node.uid) form; only nonzero cells
            in the incidence matrix will be updated by dictionary.
        aggregateby : str, optional, {None, 'last', count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'}, default : 'count'
            Method to aggregate weights of duplicate rows in data. If None, then all cell weights
            will be set to 1.
        index : bool, optional

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse matrix representation of incidence matrix for two levels of static entity.

        Note
        ----
        In the context of hypergraphs think level1 = edges, level2 = nodes
        """
        if self.dimsize < 2:
            warnings.warn("Incidence matrix requires two levels of data.")
            return None
        if not weights:  # transpose from the beginning
            if self.dimsize > 2:
                temp, _ = remove_row_duplicates(self.data[:, [level2, level1]])
            else:
                temp = self.data[:, [level2, level1]]
            result = csr_matrix((np.ones(len(temp)), temp.transpose()), dtype=int)
        else:  # transpose after cell weights are added
            if self.dimsize > 2:
                temp, temp_weights = remove_row_duplicates(
                    self.data[:, [level1, level2]],
                    weights=self._weights,
                    aggregateby=aggregateby,
                )
            else:
                temp, temp_weights = self.data[:, [level1, level2]], self.cell_weights

            if isinstance(weights, dict):
                cat1 = self.keys[level1]
                cat2 = self.keys[level2]
                for k, v in weights:
                    try:
                        tdx = (self.index(cat1, k[0]), self.index(cat2, k[1]))
                    except:
                        HyperNetXError(
                            f"{k} is not recognized as belonging to this system."
                        )
                    if temp_weights[tdx] != 0:
                        temp_weights[tdx] = v
                # weights = {(self.index(cat1, k[0]), self.index(cat2, k[1])): v for k, v in weights.items()}
                # for k in weights:
                #     if temp_weights[k] != 0::
                #         temp_weights[k]=weights[k]
            temp_weights = [temp_weights[tuple(t)] for t in temp]
            dtype = int if aggregateby == "count" else float
            result = csr_matrix(
                (temp_weights, temp.transpose()), dtype=dtype
            ).transpose()

        if index:  # give index of rows then columns
            return (
                result,
                {k: v for k, v in enumerate(self._labs[level2])},
                {k: v for k, v in enumerate(self._labs[level1])},
            )
        else:
            return result

    def restrict_to_levels(self, levels, weights=False, aggregateby="count", uid=None):
        """
        Limit Static Entity data to specific levels

        Parameters
        ----------
        levels : array
            index of labels in data
        weights : bool, optional, default : False
            Whether or not to aggregate existing weights in self when restricting to levels.
            If False then weights will be assigned 1.
        aggregateby : str, optional, {None, 'last', count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'}, default : 'count'
            Method to aggregate cell_weights of duplicate rows in setsystem of type pandas.DataFrame.
            If None then all cell_weights will be set to 1.
        uid : None, optional

        Returns
        -------
        Static Entity class
            hnx.classes.staticentity.StaticEntity
        """
        if levels[0] >= self.dimsize:
            return self.__class__()
        # if len(levels) == 1:
        #     if levels[0] >= self.dimsize:
        #         return self.__class__()
        #     else:
        #         newlabels = OrderedDict(
        #             [(self.keys[lev], self._labs[lev]) for lev in levels]
        #         )
        # return self.__class__(labels=newlabels)
        else:
            if weights:
                weights = self._weights
            else:
                weights = None
            if len(levels) == 1:
                lev = levels[0]
                newlabels = OrderedDict([(self._keys[lev], self._labs[lev])])
                data = self.data[:, lev]
                data = np.reshape(data, (len(data), 1))
                return StaticEntity(
                    data=data,
                    weights=weights,
                    aggregateby=aggregateby,
                    labels=newlabels,
                    uid=uid,
                )
            else:
                data = self.data[:, levels]
                newlabels = OrderedDict(
                    [(self.keys[lev], self._labs[lev]) for lev in levels]
                )
                return self.__class__(
                    data=data,
                    weights=weights,
                    aggregateby=aggregateby,
                    labels=newlabels,
                    uid=uid,
                )

    def turn_entity_data_into_dataframe(
        self, data_subset
    ):  # add option to include multiplicities stored in properties
        """
        Convert rows of original data in StaticEntity to dataframe

        Parameters
        ----------
        data : numpy.ndarray
            Subset of the rows in the original data held in the StaticEntity

        Returns
        -------
        pandas.core.frame.DataFrame
            Columns and cell entries are derived from data and self.labels
        """
        df = pd.DataFrame(data=data_subset, columns=self.keys)
        width = data_subset.shape[1]
        for ddx, row in enumerate(data_subset):
            nrow = [self.labs(idx)[row[idx]] for idx in range(width)]
            df.iloc[ddx] = nrow
        return df

    def restrict_to_indices(
        self, indices, level=0, uid=None
    ):  # restricting to indices requires renumbering the labels.
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
        indices = list(indices)
        idx = np.concatenate(
            [np.argwhere(self.data[:, level] == k) for k in indices], axis=0
        ).transpose()[0]
        temp = self.data[idx]
        df = self.turn_entity_data_into_dataframe(temp)
        return self.__class__(entity=df, uid=uid)

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
         : numpy.array(str)
        """
        if isinstance(index, int):
            return self._labs[level][index]
        else:
            return [self._labs[level][idx] for idx in index]

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
        assert len(coords) == self.dimsize
        translation = list()
        for idx in range(self.dimsize):
            translation.append(self.translate(idx, coords[idx]))
        return translation

    def index(self, category, value=None):
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
        if value is not None:
            return self._keyindex[category], self._index[category][value]
        else:
            return self._keyindex[category]

    def indices(self, category, values):
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
        return [self._index[category][value] for value in values]

    def level(self, item, min_level=0, max_level=None, return_index=True):
        """
        Returns first level item appears by order of keys from minlevel to maxlevel
        inclusive

        Parameters
        ----------
        item : string
        min_level : int, optional
        max_level : int, optional

        return_index : bool, optional

        Returns
        -------
        tuple
        """
        n = len(self.dimensions)
        if max_level is not None:
            n = min([max_level + 1, n])
        for lev in range(min_level, n):
            if item in self._labs[lev]:
                if return_index:
                    return lev, self._index[self._keys[lev]][item]
                else:
                    return lev
        else:
            print(f'"{item}" not found')
            return None

    # note the depth and registry methods may or may not be useful. We can add these later.


class StaticEntitySet(StaticEntity):

    """
    .. _staticentityset:
    """

    def __init__(
        self,
        entity=None,
        data=None,
        arr=None,
        labels=None,
        uid=None,
        level1=0,
        level2=1,
        weights=None,
        keep_weights=True,
        aggregateby=None,
        **props,
    ):

        if entity is None:
            if data is not None:
                data = data[:, [level1, level2]]
                arr = None
            elif arr is not None:
                data, cell_weights = _turn_tensor_to_data(arr)
                weights = [cell_weights[tuple(t)] for t in data]
                data = data[:, [level1, level2]]
            if labels is not None:
                keys = np.array(list(labels.keys()))
                temp = OrderedDict()
                for lev in [level1, level2]:
                    if lev < len(keys):
                        temp[keys[lev]] = labels[keys[lev]]
                labels = temp
            super().__init__(
                data=data, weights=weights, labels=labels, uid=uid, **props
            )
        else:
            if isinstance(entity, StaticEntity):
                data = entity.data[:, [level1, level2]]
                if keep_weights:
                    weights = entity._weights
                labels = OrderedDict(
                    [(entity._keys[k], entity._labs[k]) for k in [level1, level2]]
                )
                super().__init__(
                    data=data,
                    labels=labels,
                    uid=uid,
                    weights=weights,
                    aggregateby=aggregateby,
                    **props,
                )
            elif isinstance(entity, StaticEntitySet):
                if keep_weights:
                    aggregateby = "last"
                super().__init__(
                    entity,
                    weights=weights,
                    keep_weights=keep_weights,
                    aggregateby=aggregateby,
                    **props,
                )

            elif isinstance(entity, pd.DataFrame):
                cols = entity.columns[[level1, level2]]
                super().__init__(
                    entity=entity[cols],
                    uid=uid,
                    weights=weights,
                    aggregateby=aggregateby,
                    **props,
                )
            else:
                # this presumes entity is an iterable of iterables or a dictionary
                super().__init__(entity=entity, uid=uid, **props)

    def __repr__(self):
        """
        Returns a string resembling the constructor for entityset without any
        children

        Returns
        -------
        string
        """
        return f"StaticEntitySet({self._uid},{list(self.uidset)},{self.properties})"

    def incidence_matrix(self, index=False, weights=False):
        """
        Incidence matrix of StaticEntitySet

        Parameters
        ----------
        index : bool, optional

        weight: bool, dict optional, default=False
            If False all nonzero entries are 1.
            If True all nonzero entries are filled by self.cell_weight
            dictionary values.
            If dict, keys must be in self.cell_weight keys; nonzero cells
            will be updated by dictionary.


        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Sparse matrix representation of incidence matrix for static entity set.
        """
        return StaticEntity.incidence_matrix(self, weights=weights, index=index)

    def restrict_to(self, indices, uid=None):
        """
        Limit Static Entityset data to specific indices of keys

        Parameters
        ----------
        indices : array
            array of indices in keys
        uid : None, optional

        Returns
        -------
        StaticEntitySet
            hnx.classes.staticentity.StaticEntitySet

        """
        return self.restrict_to_indices(indices, level=0, uid=uid)

    def convert_to_entityset(self, uid):
        """
        Convert Static EntitySet into EntitySet with given uid.

        Parameters
        ----------
        uid : string

        Returns
        -------
        EntitySet
            hnx.classes.entity.EntitySet
        """
        return EntitySet(uid, self.incidence_dict)

    def collapse_identical_elements(
        self,
        uid=None,
        return_equivalence_classes=False,
    ):
        """
        Returns StaticEntitySet after collapsing elements if they have same children
        If no elements share same children, a copy of the original StaticEntitySet is returned

        Parameters
        ----------
        uid : None, optional
        return_equivalence_classes : bool, optional
            If True, return a dictionary of equivalence classes keyed by new edge names


        Returns
        -------
        StaticEntitySet
            hnx.classes.staticentity.StaticEntitySet
        """
        shared_children = DefaultOrderedDict(list)
        for k, v in self.elements.items():
            shared_children[frozenset(v)].append(k)
        new_entity_dict = OrderedDict(
            [
                # (
                #     f"{next(iter(v))}:{len(v)}",
                #     sorted(set(k), key=lambda x: list(self.labs(1)).index(x)),
                # )
                (
                    f"{next(iter(v))}:{len(v)}",
                    sorted(set(k), key=lambda x: self.index(self._keys[1], x)),
                )
                for k, v in shared_children.items()
            ]
        )
        if return_equivalence_classes:
            eq_classes = OrderedDict(
                [
                    (
                        f"{next(iter(v))}:{len(v)}",
                        v
                        # sorted(v, key=lambda x: self.index(self._keys[0], x)),  ## not sure why sorting is important here
                    )
                    for k, v in shared_children.items()
                ]
            )
            return StaticEntitySet(uid=uid, entity=new_entity_dict), eq_classes
        else:
            return StaticEntitySet(uid=uid, entity=new_entity_dict)


def _turn_tensor_to_data(arr):
    """
    Return list of nonzero coordinates in arr.

    Parameters
    ----------
    arr : numpy.ndarray
        Tensor corresponding to incidence of co-occurring labels.
    """
    temp = np.array(arr.nonzero()).T
    return temp, {tuple(t): arr[tuple(t)] for t in temp}


def _turn_dict_to_staticentity(dict_object):
    """Create a static entity directly from a dictionary of hashables"""
    d = OrderedDict(dict_object)
    level2ctr = HNXCount()
    level1ctr = HNXCount()
    level2 = DefaultOrderedDict(level2ctr)
    level1 = DefaultOrderedDict(level1ctr)
    coords = list()
    for k, val in d.items():
        level1[k]
        for v in val:
            level2[v]
            coords.append((level1[k], level2[v]))
    coords, counts = remove_row_duplicates(coords, aggregateby="count")
    level1 = np.array(list(level1))
    level2 = np.array(list(level2))
    data = np.array(coords, dtype=int)
    labels = OrderedDict({"0": level1, "1": level2})
    return data, labels, counts


def _turn_iterable_to_staticentity(iter_object):
    for s in iter_object:
        if not isinstance(s, Iterable):
            raise HyperNetXError(
                "The entity data type not recognized. Iterables must be iterable of iterables."
            )
    else:
        labels = [f"e{str(x)}" for x in range(len(iter_object))]
        dict_object = dict(zip(labels, iter_object))
    return _turn_dict_to_staticentity(dict_object)


def _turn_dataframe_into_entity(
    df, weights=None, aggregateby=None, include_unknowns=False
):
    """
    Convenience method to reformat dataframe object into data,labels format
    for construction of a static entity

    Parameters
    ----------
    df : pandas.DataFrame
        May not contain nans
    weights : array-like, optional, default : None
        User specified weights corresponding to data, length must equal number
        of rows in data. If None, weight for all rows is assumed to be 1.
    aggregateby : str, optional, {None, 'last', count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'}, default : 'count'
        Method to aggregate cell_weights of duplicate rows in data.
    include_unknowns : bool, optional, default : False
        If Unknown <column name> was used to fill in nans

    Returns
    -------
    outputdata : numpy.ndarray
    slabels : numpy.array of strings
    cell_weights : dict

    """
    columns = df.columns
    ctr = [HNXCount() for c in range(len(columns))]
    ldict = OrderedDict()
    rdict = OrderedDict()
    for idx, c in enumerate(columns):
        ldict[c] = defaultdict(ctr[idx])  # TODO make this an Ordered default dict
        rdict[c] = OrderedDict()
        if include_unknowns:
            ldict[c][f"Unknown {c}"]
            # TODO: update this to take a dict assign for each column
            rdict[c][0] = f"Unknown {c}"
        for k in df[c]:
            ldict[c][k]
            rdict[c][ldict[c][k]] = k
        ldict[c] = dict(ldict[c])
    dims = tuple([len(ldict[c]) for c in columns])

    m = len(df)
    n = len(columns)
    data = np.zeros((m, n), dtype=int)
    for rid in range(m):
        for cid in range(n):
            c = columns[cid]
            data[rid, cid] = ldict[c][df.iloc[rid][c]]

    output_data = remove_row_duplicates(data, weights=weights, aggregateby=aggregateby)

    slabels = OrderedDict()
    for cdx, c in enumerate(columns):
        slabels.update({c: np.array(list(ldict[c].keys()))})
    return output_data[0], slabels, output_data[1]


# helpers
def _fd():
    return None
