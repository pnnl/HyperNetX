from collections import OrderedDict, defaultdict
from collections.abc import Iterable
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import *
from hypernetx.exception import HyperNetXError
from hypernetx.classes.entity import Entity, EntitySet
from hypernetx.utils import HNXCount, DefaultOrderedDict, remove_row_duplicates
from scipy.sparse import coo_matrix, csr_matrix, issparse
import itertools as it
import pandas as pd

__all__ = [
    'StaticEntity',
    'StaticEntitySet'
]

class StaticEntity(object):

    """
    .. _staticentity:

    arr = np.ndarray (there can be no empty cells) of boolean or integer values
    labels = OrderedDict of labelnames by dimension, keys = header names , ## rdict with numeric keys
    levels are given by the order of these labels

    Parameters
    ----------
    arr : numpy.ndarray or scip.sparse.matrix, optional, default=None

    labels : OrderedDict of lists, optional, default=None
        dictionary lists

    entity : hypernetx.StaticEntity or hypernets.StaticEntitySet, optional, default=None

    uid : hashable, optional, default=None

    keep_state_dict : bool, optional, default=False

    props : user defined keyword arguments, optional

    """

    def __init__(self,
                 entity=None,
                 data=None,
                 arr=None,
                 labels=None,
                 uid=None,
                 **props):

        self._uid = uid
        self.properties = {}
        if entity is not None:
            if isinstance(entity, StaticEntity) or isinstance(entity, StaticEntitySet):
                self.properties.update(entity.properties)
                self.properties.update(props)
                self.__dict__.update(self.properties)
                self.__dict__.update(props)
                self._data = entity.data.copy()
                self._dimensions = entity.dimensions
                self._dimsize = entity.dimsize
                self._labels = OrderedDict((category, np.array(values)) for category, values in entity.labels.items())
                self._keys = np.array(list(self._labels.keys()))
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
                self._arr = None
            elif isinstance(entity, pd.DataFrame):
                self.properties.update(props)
                data, labels, counts = _turn_dataframe_into_entity(entity, return_counts=True)
                self.properties.update({'counts': counts})
                self.__dict__.update(self.properties)
                self._data = data
                self._labels = labels
                self._arr = None
                self._dimensions = tuple([max(x) + 1 for x in self._data.transpose()])
                self._dimsize = len(self._dimensions)
                self._keys = np.array(list(self._labels.keys()))
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
            else:
                if isinstance(entity, Entity) or isinstance(entity, EntitySet):
                    d = entity.incidence_dict
                    self._data, self._labels = _turn_dict_to_staticentity(d)  # For now duplicate entries will be removed.
                elif isinstance(entity, dict):  # returns only 2 levels
                    self._data, self._labels = _turn_dict_to_staticentity(entity)  # For now duplicate entries will be removed.
                else:  # returns only 2 levels
                    self._data, self._labels = _turn_iterable_to_staticentity(entity)
                self._dimensions = tuple([len(self._labels[k]) for k in self._labels])
                self._dimsize = len(self._dimensions)
                self._keys = np.array(list(self._labels.keys()))
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
                self.properties.update(props)
                self.__dict__.update(self.properties)  # Add function to set attributes ###########!!!!!!!!!!!!!
                self._arr = None
        elif data is not None:
            self._arr = None
            self._data, counts = remove_row_duplicates(data, return_counts=True)
            self.properties['counts'] = counts
            self._dimensions = tuple([max(x) + 1 for x in self._data.transpose()])
            self._dimsize = len(self._dimensions)
            self.properties.update(props)
            self.__dict__.update(props)
            if labels is not None:  # determine if hashmaps might be better than lambda expressions to recover indices
                self._labels = OrderedDict((category, np.array(values)) for category, values in labels.items())  # OrderedDict(category,np.array([categorical values ....])) is aligned to arr
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._keys = np.array(list(labels.keys()))
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
            else:
                self._labels = OrderedDict([(int(dim), np.arange(ct)) for dim, ct in enumerate(self.dimensions)])
                self._keyindex = lambda category: int(category)
                self._keys = np.arange(self._dimsize)
                self._index = lambda category, value: value if value in self._labels[category] else None
                # self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
        elif arr is not None:
            self._arr = arr
            self.properties.update(props)
            self.__dict__.update(props)
            self._state_dict = {'arr': arr * 1}
            self._dimensions = arr.shape
            self._dimsize = len(arr.shape)
            self._data = _turn_tensor_to_data(arr * 1)
            if labels is not None:  # determine if hashmaps might be better than lambda expressions to recover indices
                self._labels = OrderedDict((category, np.array(values)) for category, values in labels.items())
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._keys = np.array(list(labels.keys()))
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
            else:
                self._labels = OrderedDict([(int(dim), np.arange(ct)) for dim, ct in enumerate(self.dimensions)])
                self._keyindex = lambda category: int(category)
                self._keys = np.arange(self._dimsize)
                self._index = lambda category, value: value if value in self._labels[category] else None
        else:  # no entity, data or arr is given

            if labels is not None:
                self._labels = OrderedDict((category, np.array(values)) for category, values in labels.items())
                self._dimensions = tuple([len(labels[k]) for k in labels])
                self._data = np.zeros((0, len(labels)), dtype=int)
                self._arr = np.empty(self._dimensions, dtype=int)
                self._state_dict = {'arr': np.empty(self.dimensions, dtype=int)}
                self._dimsize = len(self._dimensions)
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._keys = np.array(list(labels.keys()))
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
            else:
                self._data = np.zeros((0, 0), dtype=int)
                self._arr = np.array([], dtype=int)
                self._labels = OrderedDict([])
                self._dimensions = tuple([])
                self._dimsize = 0
                self._keyindex = lambda category: None
                self._keys = np.array([])
                self._index = lambda category, value: None

            # if labels is a list of categorical values, then change it into an
            # ordered dictionary?
            self.properties = props
            self.__dict__.update(props)            # keyed by the method name and signature

        if len(self._labels) > 0:
            self._labs = lambda kdx: self._labels.get(self._keys[kdx], {})
        else:
            self._labs = lambda kdx: {}

    @ property
    def arr(self):
        if self._arr is not None:
            if type(self._arr) == int and self._arr == 0:
                print('arr cannot be computed')
        else:
            try:
                imat = np.zeros(self.dimensions, dtype=int)
                for d in self._data:
                    imat[tuple(d)] = 1
                self._arr = imat
            except Exception as ex:
                print(ex)
                print('arr cannot be computed')
                self._arr = 0
        return self._arr  # Do we need to return anything here

    @property
    def array_with_counts(self):
        if self._arr is not None:
            if type(self._arr) == int and self._arr == 0:
                print('arr cannot be computed')
            else:
                try:
                    imat = np.zeros(self.dimensions, dtype=int)
                    for d in self._data:
                        imat[tuple(d)] += 1
                    self._arr = imat
                except Exception as ex:
                    print(ex)
                    print('arr cannot be computed')
                    self._arr = 0
        return self._arr

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def dimsize(self):
        return self._dimsize

    @ property
    def keys(self):
        return self._keys

    @property
    def keyindex(self):
        return self._keyindex

    @ property
    def uid(self):
        return self._uid

    @ property
    def uidset(self):
        return self.uidset_by_level(0)

    @ property
    def elements(self):
        '''
        level1 = elements, level2 = children
        '''
        if len(self._keys) == 1:
            return {k: {} for k in self._labels[self._keys[0]]}
        else:
            return self.elements_by_level(0, translate=True)

    @ property
    def children(self):
        return set(self._labs(1))

    @ property
    def incidence_dict(self):
        return self.elements_by_level(0, 1, translate=True)

    @property
    def dataframe(self):
        return self.turn_entity_data_into_dataframe(self.data)

    def __len__(self):
        """Returns the number of elements in staticentity"""
        return self._dimensions[0]

    def __str__(self):
        """Return the staticentity uid."""
        return f'{self.uid}'

    def __repr__(self):
        """Returns a string resembling the constructor for staticentity without any
        children"""
        return f'StaticEntity({self._uid},{list(self.uidset)},{self.properties})'

    def __contains__(self, item):
        """
        Defines containment for StaticEntity based on labels/categories.
        """
        return item in np.concatenate(list(self._labels.values()))

    def __getitem__(self, item):
        # return self.elements_by_level(0, 1)[item]
        return self.elements[item]

    def __iter__(self):
        return iter(self.elements)

    def __call__(self, label_index=0):
        return iter(self._labs(label_index))

    def size(self):
        '''The number of elements in E, the size of dimension 0 in the E.arr'''
        return len(self)

    def labs(self, kdx):
        """Retrieve labels by index in keys"""
        return self._labs(kdx)

    def is_empty(self, level=0):
        """Boolean indicating if entity.elements is empty"""
        return len(self._labs(level)) == 0

    def uidset_by_level(self, level=0):
        '''The labels found in columns = level'''
        return frozenset(self._labs(level))  # should be update this to tuples?

    def elements_by_level(self, level1=0, level2=None, translate=False):
        '''
        think: level1 = edges, level2 = nodes
        Is there a better way to view a slice of self._arr?
        '''
        if level1 > self.dimsize - 1 or level1 < 0:
            print(f'This StaticEntity has no level {level1}.')
            return
        if level2 is None:
            level2 = level1 + 1

        if level2 > self.dimsize - 1 or level2 < 0:
            print(f'This StaticEntity has no level {level2}.')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])
        elif level1 == level2:
            print(f'level1 equals level2')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])

        temp = remove_row_duplicates(self.data[:, [level1, level2]])
        elts = defaultdict(list)
        for row in temp:
            elts[row[0]].append(row[1])

        if translate:
            telts = OrderedDict()
            for kdx, vec in elts.items():
                k = self._labs(level1)[kdx]
                telts[k] = list()
                for vdx in vec:
                    telts[k].append(self._labs(level2)[vdx])
            return telts
        else:
            return elts

    def incidence_matrix(self, level1=0, level2=1, weighted=False, index=False):
        '''
        Convenience method to navigate large tensor.
        level1 indexes the columns and level2 indexes the rows
        think level1 = edges, level2 = nodes
        '''
        if not weighted:
            temp = remove_row_duplicates(self.data[:, [level2, level1]])
        else:
            temp = self.data[:, [level2, level1]]
        result = csr_matrix((np.ones(len(temp)), temp.transpose()), dtype=int)

        if index:  # give index of rows then columns
            return result, {k: v for k, v in enumerate(self._labs(level2))}, {k: v for k, v in enumerate(self._labs(level1))}
        else:
            return result

    def restrict_to_levels(self, levels, uid=None):
        if len(levels) == 1:
            if levels[0] >= self.dimsize:
                return self.__class__()
            else:
                newlabels = OrderedDict([(self.keys[lev], self._labs(lev)) for lev in levels])
                return self.__class__(labels=newlabels)
        temp = remove_row_duplicates(self.data[:, levels])
        newlabels = OrderedDict([(self.keys[lev], self._labs(lev)) for lev in levels])
        return self.__class__(data=temp, labels=newlabels, uid=uid)

    def turn_entity_data_into_dataframe(self, data_subset):  # add option to include multiplicities stored in properties
        """Summary

        Parameters
        ----------
        data : numpy.ndarray
            Subset of the rows in the original data held in the StaticEntity

        Returns
        -------
        pandas.Dataframe
            Columns and cell entries are derived from data and self.labels
        """
        df = pd.DataFrame(data=data_subset, columns=self.keys)
        width = data_subset.shape[1]
        for ddx, row in enumerate(data_subset):
            nrow = [self.labs(idx)[row[idx]] for idx in range(width)]
            df.iloc[ddx] = nrow
        return df

    def restrict_to_indices(self, indices, level=0, uid=None):  # restricting to indices requires renumbering the labels.

        indices = list(indices)
        idx = np.concatenate([np.argwhere(self.data[:, level] == k) for k in indices], axis=0).transpose()[0]
        temp = self.data[idx]
        df = self.turn_entity_data_into_dataframe(temp)
        return self.__class__(entity=df, uid=uid)

    def translate(self, level, index):
        # returns category of dimension and value of index in that dimension
        if isinstance(index, int):
            return self._labs(level)[index]
        else:
            return [self._labs(level)[idx] for idx in index]

    def translate_arr(self, coords):
        '''Translates a single cell in the entity array'''
        assert len(coords) == self.dimsize
        translation = list()
        for idx in range(self.dimsize):
            translation.append(self.translate(idx, coords[idx]))
        return translation

    def index(self, category, value=None):
        # returns dimension of category and index of value
        if value is not None:
            return self._keyindex(category), self._index(category, value)
        else:
            return self._keyindex(category)

    def indices(self, category, values):
        return [self._index(category, value) for value in values]

    def level(self, item, min_level=0, max_level=None, return_index=True):
        '''returns first level item appears by order of keys from minlevel to maxlevel
        inclusive'''
        n = len(self.dimensions)
        if max_level is not None:
            n = min([max_level + 1, n])
        for lev in range(min_level, n):
            if item in self._labs(lev):
                if return_index:
                    return lev, self._index(self._keys[lev], item)
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

    def __init__(self,
                 entity=None,
                 data=None,
                 arr=None,
                 labels=None,
                 uid=None,
                 level1=0,
                 level2=1,
                 **props):

        if entity is None:
            if data is not None:
                data = data[:, [level1, level2]]
                arr = None
            elif arr is not None:
                data = _turn_tensor_to_data(arr)
                data = data[:, [level1, level2]]
                arr = None
            if labels is not None:
                keys = np.array(list(labels.keys()))
                temp = OrderedDict()
                for lev in [level1, level2]:
                    if lev < len(keys):
                        temp[keys[lev]] = labels[keys[lev]]
                labels = temp
            super().__init__(data=data,
                             arr=arr,
                             labels=labels,
                             uid=uid,
                             **props)
        else:
            E = StaticEntity(entity=entity)
            E = E.restrict_to_levels([level1, level2])
            super().__init__(entity=E, uid=uid, **props)

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f'StaticEntitySet({self._uid},{list(self.uidset)},{self.properties})'

    def incidence_matrix(self, sparse=True, weighted=False, index=False):
        if not weighted:
            temp = remove_row_duplicates(self.data[:, [1, 0]])
        else:
            temp = self.data[:, [1, 0]]
        result = csr_matrix((np.ones(len(temp)), temp.transpose()), dtype=int)

        if index:  # give index of rows then columns
            return result, {k: v for k, v in enumerate(self._labs(1))}, {k: v for k, v in enumerate(self._labs(0))}
        else:
            return result

    def restrict_to(self, indices, uid=None):
        return self.restrict_to_indices(indices, level=0, uid=uid)

    def convert_to_entityset(self, uid):
        return(hnx.EntitySet(uid, self.incidence_dict))

    def collapse_identical_elements(self, uid=None, return_equivalence_classes=False,):
        shared_children = DefaultOrderedDict(list)
        for k, v in self.elements.items():
            shared_children[frozenset(v)].append(k)
        new_entity_dict = OrderedDict([(f"{next(iter(v))}:{len(v)}", sorted(set(k), key=lambda x: list(self.labs(1)).index(x))) for k, v in shared_children.items()])
        if return_equivalence_classes:
            eq_classes = OrderedDict([(f"{next(iter(v))}:{len(v)}", sorted(v, key=lambda x: list(self.labs(0)).index(x))) for k, v in shared_children.items()])
            return StaticEntitySet(uid=uid, entity=new_entity_dict), eq_classes
        else:
            return StaticEntitySet(uid=uid, entity=new_entity_dict)


def _turn_tensor_to_data(arr, remove_duplicates=True):
    """
    Return list of nonzero coordinates in arr.

    Parameters
    ----------
    arr : numpy.ndarray
        Tensor corresponding to incidence of co-occurring labels.
    """
    return np.array(arr.nonzero()).transpose()
    # dfa = np.array(arr.nonzero()).transpose()
    # if remove_duplicates:
    #     return remove_row_duplicates(dfa)
    # else:
    #     return dfa


def _turn_dict_to_staticentity(dict_object, remove_duplicates=True):
    '''Create a static entity directly from a dictionary of hashables'''
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
    if remove_duplicates:
        coords = remove_row_duplicates(coords)
    level1 = list(level1)
    level2 = list(level2)
    data = np.array(coords, dtype=int)
    labels = OrderedDict({'0': level1, '1': level2})
    return data, labels


def _turn_iterable_to_staticentity(iter_object, remove_duplicates=True):
    for s in iter_object:
        if not isinstance(s, Iterable):
            raise HyperNetXError('StaticEntity constructor requires an iterable of iterables.')
    else:
        labels = [f'e{str(x)}' for x in range(len(iter_object))]
        dict_object = dict(zip(labels, iter_object))
    return _turn_dict_to_staticentity(dict_object, remove_duplicates=remove_duplicates)


def _turn_dataframe_into_entity(df, return_counts=False, include_unknowns=False):
    """
    Convenience method to reformat dataframe object into data,labels format
    for construction of a static entity 

    Parameters
    ----------
    df : pandas.DataFrame
        May not contain nans
    return_counts : bool, optional, default : False
        Used for keeping weights
    include_unknowns : bool, optional, default : False
        If Unknown <column name> was used to fill in nans

    Returns
    -------
    outputdata : numpy.ndarray
    counts : numpy.array of ints
    slabels : numpy.array of strings

    """
    columns = df.columns
    ctr = [HNXCount() for c in range(len(columns))]
    ldict = OrderedDict()
    rdict = OrderedDict()
    for idx, c in enumerate(columns):
        ldict[c] = defaultdict(ctr[idx])  # TODO make this an Ordered default dict
        rdict[c] = OrderedDict()
        if include_unknowns:
            ldict[c][f'Unknown {c}']  # TODO: update this to take a dict assign for each column
            rdict[c][0] = f'Unknown {c}'
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

    output_data = remove_row_duplicates(data, return_counts=return_counts)

    slabels = OrderedDict()
    for cdx, c in enumerate(columns):
        slabels.update({c: np.array(list(ldict[c].keys()))})
    if return_counts:
        return output_data[0], slabels, output_data[1]
    else:
        return output_data, slabels
