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
    arr = np.ndarray (there can be no empty cells) of boolean values
    labels = OrderedDict of labelnames by dimension, keys = header names , ## rdict with numeric keys
    levels are given by the order of these labels
    """

    def __init__(self, arr, labels=None, uid='test', **props):

        self._uid = uid
        self._dims = arr.shape

        if labels:
            self._labels = labels  # rdict. ## enforce this to be an OrderedDict
            self._headers = np.array(list(labels.keys()))
            rlabels = defaultdict(dict)
            for e in self._labels:
                for k, v in self._labels[e].items():
                    rlabels[e][v] = k
            self._rlabels = rlabels

        else:
            self._labels = self._rlabels = OrderedDict([(dim, np.arange(ct)) for dim, ct in enumerate(self._dims)])
            self._headers = np.arange(len(self._dims))

        self._labs = lambda kdx: self._labels.get(self._headers[kdx], {})
        self._arr = arr * 1  # imat = tensor of values

        self._properties = props
        self.__dict__.update(props)

        self.state_dict = dict()  # add a decorator to each method which stores the result in the statedict
        # keyed by the method name and signature

        # This will include the NaN Value if there is one -
        # may want to include a dropna option
        self.dimensions = self._arr.shape

    @property
    def labels(self):
        return self._labels

    @property
    def reverse_labels(self):
        return self._rlabels
    

    @property
    def headers(self):
        return self._headers

    @property
    def properties(self):
        return self._properties

    @property
    def uid(self):
        return self._uid

    @property
    def elements(self):
        '''
        level1 = elements, level2 = children
        '''
        return self.elements_by_level(0, 1)

    @property
    def children(self):
        return set(self._labs(1).keys())

    def is_empty(self, level=0):
        """Boolean indicating if entity.elements is empty"""
        return len(self._labs(level)) == 0

    def uidset(self, level=0):
        return frozenset(self._labs(level).keys())

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
            elts = OrderedDict([[k, {}] for k in self._labs(level1)])
        if level1 == level2:
            print(f'level1 must be different than level2')
            elts = OrderedDict([[k, {}] for k in range(10)])

        if issparse(self._arr):
            elts = OrderedDict()
            for e in self._labs(0):
                elts[e] = self._arr.getcol(e).nonzero()[0].astype(int)
        else:
            mat = self.incidence_matrix(level1, level2)
            elts = OrderedDict([[kdx, np.where(mat[:, kdx] != 0)[0]] for kdx in range(self.dimensions[level1])])
        if translate:
            return {self.translate(kdx, level1): [self.translate(vdx, level2) for vdx in v] for kdx, v in elts.items()}
        else:
            return elts

    def incidence_matrix(self, level1=0, level2=1, weighted=False, index=False):
        '''
        add decorator which returns a partial result to the function and then continues
        think level1 = edges, level2 = nodes
        '''
        # Use state dictionary here to retrieve result if already computed
        # otherwise store result in static dictionary
        if not issparse(self._arr):
            axes = [x for x in range(len(self._labels)) if x not in [level1, level2]]
            order = [level2, level1] + axes
            temp = self._arr.transpose(tuple(order))
            result = np.sum(temp, axis=tuple(range(2, len(self.dimensions))))  # this is the weighted incidence matrix - can we do this with sparse matrices?
        else:
            if level1 == 0 and level2 == 1:
                result = self._arr.transpose()
            else:
                result = self._arr

        if not weighted:
            result = result.astype(bool) * 1
        if index:
            return result, self._labs(level2), self._labs(level1)
        else:
            return result

    def translate(self, idx, level=0):
        return self._labs(level)[idx]

    def level(self, item, min_level=0, max_level=None, return_index=True):
        n = len(self._dims)
        if max_level:
            n = min([max_level + 1, n])

        for lev in range(min_level, n):
            if item in self._labs(lev).values():
                if return_index:
                    return lev, self._rlabels[self._headers[lev]][item]
                else:
                    return lev
        else:
            print(f'"{item}" not found')
            return None

    # note the depth and registry methods may or may not be useful. We can add these later.

class StaticEntitySet(StaticEntity):

    def __init__(self, arr, labels=None, uid='test', **props):
