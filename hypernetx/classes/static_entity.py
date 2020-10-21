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
    """

    def __init__(self, arr, labels=None, uid='test', **props):

        self._uid = uid
        self._dims = arr.shape
        self._size = len(arr.shape)
        self._arr = arr * 1  # tensor of values instead of booleans
        # if labels is a list of categorical values, then change it into an
        # ordered dictionary?

        if labels:  # determine if hashmaps might be better than lambda expressions to recover indices
            self._labels = OrderedDict((category, np.array(values)) for category, values in labels.items())  # OrderedDict(category,np.array([categorical values ....])) aligned to arr
            self._keyindex = lambda category: int(np.where(np.array(list(self.labels.keys())) == category)[0])
            self._keys = np.array(list(labels.keys()))
            self._index = lambda category, value: int(np.where(self._labels[category] == value)[0])

        else:
            self._labels = self._index = OrderedDict([(dim, np.arange(ct)) for dim, ct in enumerate(self._dims)])
            self._keyindex = lambda category: category
            self._keys = np.arange(len(self._dims))
            self._index = lambda category, value: value

        self._labs = lambda kdx: self._labels.get(self._keys[kdx], {})

        self.properties = props
        self.__dict__.update(props)

        self.state_dict = dict()  # add a decorator to each method which stores the result in the statedict
        # keyed by the method name and signature

    @property
    def arr(self):
        return self._arr

    @property
    def labels(self):
        return self._labels

    @property
    def keys(self):
        return self._keys

    @property
    def uid(self):
        return self._uid

    @property
    def uidset(self):
        return self.uidset_by_level(0)

    @property
    def dimensions(self):
        return self._dims

    @property
    def size(self):
        return self._size

    @property
    def elements(self):
        '''
        level1 = elements, level2 = children
        '''
        return self.elements_by_level(0, 1, translate=True)

    @property
    def children(self):
        return set(self._labs(1))

    def __len__(self):
        """Returns the number of elements in staticentity"""
        return self._dims[0]

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

    def __call__(self, label_index):
        return iter(self._labs(label_index))

    # def __set

    def is_empty(self, level=0):
        """Boolean indicating if entity.elements is empty"""
        return len(self._labs(level)) == 0

    def uidset_by_level(self, level=0):
        return frozenset(self._labs(level))  # should be update this to tuples?

    def incidence_dict(self, level1=0, level2=1, translate=True):
        return self.elements_by_level(level1, level2)

    def elements_by_level(self, level1=0, level2=1, translate=False):
        '''
        think: level1 = edges, level2 = nodes
        Is there a better way to view a slice of self._arr?
        '''
        if level1 > len(self._dims) - 1 or level1 < 0:
            print(f'StaticEntity has no level {level1}.')
            return
        if level2 > len(self._dims) - 1 or level1 < 0:
            print(f'StaticEntity has no level {level2}.')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])
        elif level1 == level2:
            print(f'level1 equals level2')
            elts = OrderedDict([[k, np.array([])] for k in self._labs(level1)])

        if issparse(self._arr):
            elts = OrderedDict()
            if level1 == 0 and level2 == 1:
                mat = self.incidence_matrix()
                for e in range(self._dims[0]):
                    elts[e] = mat.getcol(e).nonzero()[0].astype(int)
            elif level1 == 1 and level2 == 0:
                for e in range(self._dims[1]):
                    elts[e] = self._arr.getcol(e).nonzero()[0].astype(int)
            else:
                print('sparse matrices have only two levels. Use 0,1 or 1,0.')
        else:
            mat = self.incidence_matrix(level1, level2)
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

    def incidence_matrix(self, level1=0, level2=1, weighted=False, index=False):
        '''
        level1 indexes the columns and level2 indexes the rows
        add decorator which returns a partial result to the function and then continues
        think level1 = edges, level2 = nodes
        '''
        # Use state dictionary here to retrieve result if already computed
        # otherwise store result in static dictionary
        if not issparse(self._arr):
            if self.size > 2:
                axes = [x for x in range(len(self._labels)) if x not in [level1, level2]]
                mat = self._arr.transpose(tuple([level2, level1] + axes))
                result = np.sum(self.arr, axis=tuple(range(2, self.size)))  # this is the weighted incidence matrix - can we do this with sparse matrices?
            else:
                result = self.arr.transpose([level2, level1])
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

    def restrict_to_levels(self, levels):
        if self.size == len(levels):
            return self
        newlabels = OrderedDict([(self.keys[lev], self._labs(lev)) for lev in levels])
        axes = [x for x in range(self.size) if x not in levels]
        temp = self.arr.transpose(tuple(list(levels) + axes))
        newarr = np.sum(temp, axis=tuple(range(len(levels), self.size)))
        return self.__class__(newarr, newlabels)

    def restrict_to_indices(self, indices, level=0, **props):
        if len(indices) == len(self._labs(level)):
            return self
        indices = list(indices)
        newlabels = self.labels.copy()
        newlabels[self.keys[level]] = self._labs(level)[indices]
        if level == 0:
            newarr = self.arr[indices]
        else:
            axes = [level] + list(range(1, level)) + [0] + list(range(level + 1, self.size))
            newarr = self.arr.transpose(axes)[indices].transpose(axes)
        return self.__class__(newarr, newlabels, **props)

    def translate(self, level, index):
        # returns category of dimension and value of index in that dimension
        if isinstance(index, int):
            return self._labs(level)[index]
        else:
            return [self._labs(level)[idx] for idx in index]

    def translate_arr(self, coords):
        '''Translates a single cell in the array'''
        assert len(coords) == self.size
        translation = list()
        for idx in range(self.size):
            translation.append(self.translate(idx, coords[idx]))
        return translation

    def index(self, category, value):
        # returns dimension of category and index of value
        return self._keyindex(category), self._index(category, value)

    def level(self, item, min_level=0, max_level=None, return_index=True):
        '''returns first level item appears by order of keys from minlevel to maxlevel'''
        n = len(self._dims)
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

    def __init__(self, arr, labels=None, level1=0, level2=1, uid='test', **props):
        if len(arr.shape) > 2:
            newarr = np.sum(arr, axis=tuple([level1, level2]))
            if labels:
                newlabels = OrderedDict([(k, labels[k]) for k in np.array(list(labels.keys()))[[level1, level2]]])
        else:
            newlabels = labels
            newarr = arr
        super().__init__(newarr, labels=newlabels, uid='test', **props)

    def __repr__(self):
        """Returns a string resembling the constructor for entityset without any
        children"""
        return f'StaticEntitySet({self._uid},{list(self.uidset)},{self.properties})'

    # def collapse_identical_elements(self, use_reps=False,
    #                                     return_counts=False,
    #                                     return_equivalence_classes=False):

    #     eq_classes = defaultdict(set)
    #     for e,v in self.elements.items():
    #         eq_classes[frozenset(v)].add(e)
    #     if use_reps:
    #         if return_counts:
    #             # labels equivalence class as (rep,count) tuple
    #             new_entity_dict = {(next(iter(v)), len(v)): set(k) for k, v in eq_classes.items()}
    #         else:
    #             # labels equivalence class as rep;
    #             new_entity_dict = {next(iter(v)): set(k) for k, v in eq_classes.items()}
    #     else:
    #         new_entity_dict = {frozenset(v): set(k) for k, v in eq_classes.items()}
    #     if return_equivalence_classes:
    #         return EntitySet(newuid, new_entity_dict), dict(eq_classes)  ######### something is wrong with this!!!!!!!!
    #     return EntitySet(newuid, new_entity_dict)

    def restrict_to(self, indices):
        return self.restrict_to_indices(indices, level=0)
