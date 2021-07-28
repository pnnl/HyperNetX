from collections import OrderedDict, defaultdict
from collections.abc import Callable
import numpy as np
import pandas as pd

__all__ = [
    "HNXCount",
    "DefaultOrderedDict",
    "remove_row_duplicates",
    "create_labels",
    "reverse_dictionary"
]


class HNXCount:
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
        if default_factory is not None and not isinstance(default_factory, Callable):
            raise TypeError("first argument must be callable")
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
            args = (self.default_factory,)
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy

        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return "OrderedDefaultDict(%s, %s)" % (
            self.default_factory,
            OrderedDict.__repr__(self),
        )


def remove_row_duplicates(data, weights=None, aggregateby=None):
    """
    Wrapper for numpy's unique method.
    Removes duplicate rows in a 2d array and returns counts
    if requested

    Parameters
    ----------
    data : array_like, pandas.DataFrame
        2-dimensional array_like object or dataframe
    weights : array_like, optional, default : None
        1-dimensional array_like object, must be the same length as 0-axis of data
        If None then weights are all assigned 1.
    aggregateby : str, optional, {None, 'last', count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'}, default : None
        Method to aggregate weights of duplicate rows in data. If None, then only 
        de-duped rows will be returned

    Returns
    -------
    : numpy.ndarray
        data with duplicate rows removed
    : dict
        keyed by rows in data with aggregated weights


    """
    if aggregateby is None:
        if isinstance(data, pd.DataFrame):
            return np.unique(data.values, axis=0),
        else:
            return np.unique(data, axis=0)
    else:
        try:
            r, c = data.shape
        except:
            r, c = np.array(data).shape

        df1 = pd.DataFrame(data, columns=range(c))

        if weights is None:
            df2 = pd.DataFrame(np.ones((r, 1)), columns=[c])
        else:
            if len(weights) < r:
                raise HyperNetXError(
                    "length of weight array must match number of rows in data"
                )
            df2 = pd.DataFrame(np.array(weights).reshape(r, 1), columns=[c])

        dfc = pd.concat([df1, df2], axis=1)

        # acceptable values: 'count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'
        if aggregateby == 'count':
            G = dfc.groupby(list(df1.columns)).count()
        elif aggregateby == 'sum':
            G = dfc.groupby(list(df1.columns)).sum()
        elif aggregateby == 'mean':
            G = dfc.groupby(list(df1.columns)).mean()
        elif aggregateby == 'median':
            G = dfc.groupby(list(df1.columns)).median()
        elif aggregateby == 'max':
            G = dfc.groupby(list(df1.columns)).max()
        elif aggregateby == 'min':
            G = dfc.groupby(list(df1.columns)).min()
        elif aggregateby == 'first':
            G = dfc.groupby(list(df1.columns)).first()
        elif aggregateby == 'last':
            G = dfc.groupby(list(df1.columns)).last()

        else:
            raise HyperNetXError("Acceptable values for aggregateby are: None, 'count', 'sum', 'mean', 'median', max', 'min', 'first', 'last'")

        if c == 1:
            return G.reset_index()[df1.columns].values, {tuple([k]): v for k, v in G.to_dict()[c].items()}
        else:
            return G.reset_index()[df1.columns].values, G.to_dict()[c]


def create_labels(
    num_edges,
    num_nodes,
    edgeprefix="e",
    nodeprefix="v",
    edgelabel="Edges",
    nodelabel="Nodes",
):
    """
    Creates default labels for static entity sets derived without labels

    Parameters
    ----------
    num_edges : int

    num_nodes : int

    edgeprefix : str, optional, default : 'e'

    nodeprefix : str, optional, default : 'v'

    edgelabel : str, optional, default : 'Edges'

    nodelabel : str, optional, default : 'Nodes'


    Returns
    -------
    OrderedDict
        used for labels in constructing a StaticEntitySet
    """
    enames = np.array([f"{edgeprefix}{idx}" for idx in range(num_edges)])
    nnames = np.array([f"{nodeprefix}{jdx}" for jdx in range(num_nodes)])
    return OrderedDict([(edgelabel, enames), (nodelabel, nnames)])


def reverse_dictionary(d):
    new_d = DefaultOrderedDict(list)
    for key, values in d.items():
        for val in values:
            new_d[val].append(key)

    return new_d
