from collections import OrderedDict
from collections.abc import Callable
import numpy as np

__all__ = [
    'HNXCount',
    'DefaultOrderedDict',
    'remove_row_duplicates'
]


class HNXCount():
    def __init__(self, init=0):
        self.init = init
        self.value = init

    def __call__(self):
        temp = self.value
        self.value += 1
        return temp


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


def remove_row_duplicates(arr, return_counts=False):
    """
    Wrapper for numpy's unique method.
    Removes duplicate rows in a 2d array and returns counts
    if requested

    Parameters
    ----------
    arr : array_like, 
        2-dimensional array_like object
    return_counts : bool, optional.  #### Still to do
        Returns vector of counts ordered by output order

    Returns
    -------
    : numpy.ndarray
    : numpy.array



    """
    return np.unique(arr, axis=0, return_counts=return_counts)
