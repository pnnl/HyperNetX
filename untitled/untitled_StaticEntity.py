from collections import OrderedDict, defaultdict
import warnings
from copy import copy
import numpy as np
import networkx as nx
from hypernetx import *
from scipy.sparse import coo_matrix, issparse
import itertools as it

import pandas as pd  # do we need this import?


class StaticEntity(object):

    """
    arr = np.ndarray (there can be no empty cells) of boolean or integer values
    labels = OrderedDict of labelnames by dimension, keys = header names , ## rdict with numeric keys
    levels are given by the order of these labels

    Parameters
    ----------
    arr : numpy.ndarray or scip.sparse.matrix, optional, default=None

    labels : OrderedDict, optional, default=None
        dictionary lists

    entity : hypernetx.StaticEntity or hypernets.StaticEntitySet, optional, default=None

    uid : hashable, optional, default=None

    keep_state_dict : bool, optional, default=False

    nwhy : bool, optional, default=False

    props : user defined keyword arguments, optional

    """

    def __init__(self, arr=None,
                 labels=None,
                 entity=None,
                 uid=None,
                 keep_state_dict=False,
                 nwhy=False,
                 **props):

        self._uid = uid
        self._arr = arr
        self._labels = labels
        if entity is not None and (isinstance(entity, StaticEntity) or isinstance(entity, StaticEntitySet)):
            self._arr = entity.arr.copy()
            self._dimensions = self._arr.shape
            self._dimsize = len(self._arr.shape)

            self._labels = entity.labels.copy()
            self._keys = np.array(list(self._labels.keys()))
            self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
            self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None

            if keep_state_dict:
                self.state_dict = entity.state_dict.copy()
            else:
                self.state_dict = dict()

            self.properties = entity.properties
            self.properties.update(props)
            self.__dict__.update(self.properties)

        else:
            if arr is not None:
                self._arr = arr * 1  # tensor of values instead of booleans
                if issparse(self._arr):
                    self._arr = csr_matrix(self._arr)
                self._dimensions = self._arr.shape
                self._dimsize = len(self._arr.shape)
            else:
                if labels is not None:
                    n = len(self._labels)
                    self._arr = np.array([])
                    self._dimensions = tuple([len(labels[k]) for k in labels])
                    self._dimsize = len(self._dimensions)

            if labels is not None:  # determine if hashmaps might be better than lambda expressions to recover indices
                self._labels = OrderedDict((category, np.array(values)) for category, values in labels.items())  # OrderedDict(category,np.array([categorical values ....])) aligned to arr
                self._keyindex = lambda category: int(np.where(np.array(list(self._labels.keys())) == category)[0])
                self._keys = np.array(list(labels.keys()))
                self._index = lambda category, value: int(np.where(self._labels[category] == value)[0]) if np.where(self._labels[category] == value)[0].size > 0 else None
            else:
                if arr is not None:
                    self._labels = self._index = OrderedDict([(dim, np.arange(ct)) for dim, ct in enumerate(self.dims)])
                    self._keyindex = lambda category: category
                    self._keys = np.arange(len(self.dims))
                    self._index = lambda category, value: value

            # if labels is a list of categorical values, then change it into an
            # ordered dictionary?
            self.properties = props
            self.__dict__.update(props)

            self.state_dict = dict()  # add a decorator to each method which stores the result in the statedict
            # keyed by the method name and signature

        self._labs = lambda kdx: self._labels.get(self._keys[kdx], {})

    @ property
    def arr(self):
        return self._arr

    @ property
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

    @ property
    def uid(self):
        return self._uid

    @ property
    def uidset(self):
        return self.uidset_by_level(0)

    @ property
    def size(self):
        '''The number of elements in E, the size of dimension 0 in the E.arr'''
        return len(self)

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

    def label_index(self, labels):
        '''labels that you want the header index for'''
        return [self._keyindex(label) for label in labels]

    # def __getitem__(self, level=0):
    #     '''item is the index of a header...this might require an option for translate'''
    #     if item in self.elements:
    #         return self.elements[item]
    #     else:
    #         return ''

    # def __iter__(self, level=0):
    #     return iter(self.elements)

    def __call__(self, label_index=0):
        return iter(self._labs(label_index))

    # def __set

    def is_empty(self, level=0):
        """Boolean indicating if entity.elements is empty"""
        return len(self._labs(level)) == 0

    def uidset_by_level(self, level=0):
        return tuple(self._labs(level))  # should be update this to tuples?

    def elements_by_level(self, level1=0, level2=None, translate=False):
        '''
        think: level1 = edges, level2 = nodes
        Is there a better way to view a slice of self._arr?
        '''
        if level1 > self.dimsize - 1 or level1 < 0:
            print(f'This StaticEntity has no level {level1}.')
            return
        if not level2:
            level2 = level1 + 1

        if level2 > self.dimsize - 1 or level1 < 0:
            print(f'This StaticEntity has no level {level2}.')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])
        elif level1 == level2:
            print(f'level1 equals level2')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])

        elif issparse(self._arr):
            elts = OrderedDict()
            if level1 == 0 and level2 == 1:
                mat = self._incidence_matrix()
                for e in range(self.dimensions[0]):
                    elts[e] = mat.getcol(e).nonzero()[0].astype(int)
            elif level1 == 1 and level2 == 0:
                for e in range(self.dimensions[1]):
                    elts[e] = self._arr.getcol(e).nonzero()[0].astype(int)
            else:
                print('sparse matrices have only two levels. Use 0,1 or 1,0.')
                return
        else:
            mat = self._incidence_matrix(level1, level2)
            elts = OrderedDict([[kdx, np.where(mat[:, kdx] != 0)[0]] for kdx in range(self.dimensions[level1])])

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

    def _incidence_matrix(self, level1=0, level2=1, weighted=False, index=False):
        '''
        Convenience method to navigate large tensor.
        level1 indexes the columns and level2 indexes the rows
        add decorator which returns a partial result to the function and then continues
        think level1 = edges, level2 = nodes
        '''
        # Use state dictionary here to retrieve result if already computed
        # otherwise store result in static dictionary
        if not issparse(self._arr):
            if self.dimsize > 2:
                axes = [x for x in range(len(self._labels)) if x not in [level1, level2]]
                mat = self._arr.transpose(tuple([level2, level1] + axes))
                result = np.sum(self._arr, axis=tuple(range(2, self.dimsize)))  # this is the weighted incidence matrix - can we do this with sparse matrices?
            else:
                result = self._arr.transpose([level2, level1])
        else:
            if level1 == 0 and level2 == 1:
                result = self._arr.transpose()
            else:
                result = self._arr

        if not weighted:
            result = result.astype(bool) * 1
        if index:  # give index of rows then columns
            return result, self._labs(level2), self._labs(level1)
        else:
            return result

    def restrict_to_levels(self, levels, uid=None):
        if len(levels) == 1:
            newlabels = OrderedDict([(self.keys[lev], self._labs(lev)) for lev in levels])
            return self.__class__(labels=newlabels)
        if issparse(self._arr):
            if levels == (1, 0):
                newarr = self._arr.transpose()
            else:
                newarr = self._arr
        else:
            axes = [x for x in range(self.dimsize) if x not in levels]
            temp = self._arr.transpose(tuple(list(levels) + axes))
            newarr = np.sum(temp, axis=tuple(range(len(levels), self.dimsize)))
        newlabels = OrderedDict([(self.keys[lev], self._labs(lev)) for lev in levels])
        return self.__class__(newarr, newlabels, uid=uid)

    def remove_empties(self, level_to_keep=0):
        axes = list(np.arange(self.dimsize))
        axes.remove(level_to_keep)
        ne = self.__class__(entity=self)
        for lev in axes:
            ids = list(range(ne.dimsize))
            ids.remove(lev)
            ids = tuple(ids)
            keepers = np.sum(ne.arr, axis=ids).nonzero()[0]
            ne = ne.restrict_to_indices(keepers, level=lev, uid=ne.uid, rem_empties=False)
        return ne

    def restrict_to_indices(self, indices, level=0, uid=None, rem_empties=True):
        # TODO handle sparse case - use scipy.sparse.bmat
        indices = list(indices)
        newlabels = self.labels.copy()
        newlabels[self.keys[level]] = self._labs(level)[indices]
        if issparse(self._arr):
            newarr = self._arr.tocsr
            if level > 1:
                print('static entities with sparse arrays have at most 2 levels')
                return
            elif level == 1:
                newarr = self._arr[:, indices]
            else:
                newarr = self._arr[indices]
        else:
            if level == 0:
                newarr = self._arr[indices]
            else:
                axes = [level] + list(range(1, level)) + [0] + list(range(level + 1, self.dimsize))
                newarr = self._arr.transpose(axes)[indices].transpose(axes)
        temp = self.__class__(arr=newarr, labels=newlabels, uid=uid, **self.properties)
        # return temp
        if rem_empties == True:
            return temp.remove_empties(level)
        else:
            return temp

    def translate(self, level, index):
        # returns category of dimension and value of index in that dimension
        if isinstance(index, int):
            return self._labs(level)[index]
        else:
            return [self._labs(level)[idx] for idx in index]

    def translate_arr(self, coords):
        '''Translates a single cell in the array'''
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
        # returns dimension of category and index of value
        # results = list()
        # for val in values:
        #     try:
        #         results.append(self._index(category, val))
        #     except:
        #         print(category, val)
        #         return results
        # else:
        #     return results
        return [self._index(category, value) for value in values]

    def level(self, item, min_level=0, max_level=None, return_index=True):
        '''returns first level item appears by order of keys from minlevel to maxlevel'''
        n = len(self.dimensions)
        if max_level:
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

    def __init__(self, arr=None,
                 labels=None,
                 level1=0,
                 level2=1,
                 entity=None,
                 uid=None,
                 keep_state_dict=False,
                 **props):

        E = StaticEntity(arr=arr,
                         labels=labels,
                         entity=entity,
                         uid=uid,
                         keep_state_dict=keep_state_dict,
                         **props)
        if len(E.dimensions) > 2:
            E = E.restrict_to_levels((level1, level2))

        super().__init__(entity=E)

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f'StaticEntitySet({self._uid},{list(self.uidset)},{self.properties})'

    def incidence_matrix(self, sparse=True, index=False):

        mat = self._arr.transpose()
        if sparse:
            mat = csr_matrix(mat).astype(bool) * 1
        elif issparse(mat):
            mat = mat.todense().astype(bool) * 1

        if index:
            rdict = dict(zip(range(self.dimensions[1]), self._labs(1)))
            cdict = dict(zip(range(self.dimensions[0]), self._labs(0)))
            return mat, rdict, cdict
        else:
            return mat

    def restrict_to(self, indices, uid=None):
        return self.restrict_to_indices(indices, level=0, uid=uid)

    def convert_to_entityset(self, uid):
        return(hnx.EntitySet(uid, self.incidence_dict))

    def collapse_identical_elements(self, use_reps=False,
                                    return_counts=False,
                                    return_equivalence_classes=False):
        pass
