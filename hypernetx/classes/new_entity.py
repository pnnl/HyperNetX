from hypernetx import *
import pandas as pd
import numpy as np
from collections import OrderedDict


class Entity:
    def __init__(
        self,
        data=None,
        arr=None,
        labels=None,
    ):
        self.properties = {}

        if data is not None:
            self._init_from_data(data, labels)
        elif arr is not None:
            self._init_from_incidence(arr)
        else:
            self._init_empty()

    def _init_from_data(self, data, labels):
        """
        Initialize the Entity when explicitly given data as a Pandas DataFrame or NumPy
        ndarray (TODO: dict of data). Optional: dict of labels
        """

        if isinstance(data, pd.DataFrame):
            # convert DataFrame to encoded ndarray with labels
            self._data, labels = _data_from_dataframe(data)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self._data = data
        elif isinstance(data, dict):
            # TODO
            pass

        # dimensions of the full data tensor
        self._dimensions = tuple(self._data.max(axis=0) + 1)
        self._dimsize = self._data.shape[1]

        # assign labels
        self._assign_labels(labels)

    def _init_from_incidence(self, arr):
        """
        I think we should deprecate this - searched through algorithms and can't find
        any instances of arr being used or referenced at all. Storing a potentially
        massive incidence tensor as a dense np.ndarray seems wasteful
        """
        pass

    def _init_empty(self, labels):
        """
        Initialize the entity when no data is given. Optional: dict of labels
        """
        # assign labels
        self._assign_labels(labels)

        # dimensions are given based on the labels
        self._dimensions = tuple(len(self._labels[category] for category in self._keys))
        self._dimsize = len(self._keys)

        # empty data
        self._data = np.ndarray((0, self._dimsize), dtype=int)

    def _assign_labels(self, labels):
        """
        Fills in labels and related fields in the Entity with provided or generated
        values.
        """
        # use given labels if provided with a valid dict
        if isinstance(labels, dict):
            if all(isinstance(vals, np.ndarray) for vals in labels.values()):
                self._labels = OrderedDict(labels)
            else:
                self._labels = OrderedDict(
                    (category, np.asarray(values))
                    for category, values in labels.items()
                )
        # if dimensions of data tensor are known, labels are data tensor indices
        elif hasattr(self, "_dimensions"):
            self._labels = OrderedDict(
                (dim, np.arange(ct)) for dim, ct in enumerate(self._dimensions)
            )
        else:
            self._labels = OrderedDict()

        # construct dictionaries to translate category keys and labels to column indices
        # and encoded values in the data
        self._keys = np.array(list(self._labels.keys()))
        self._keyindex = {category: i for i, category in enumerate(self._keys)}
        # NOTE: preivously called _labs, but I found that non-descriptive
        self._kdxlabels = {
            i: self._labels[category] for category, i in self._keyindex.items()
        }
        self._index = {
            category: {key: i for i, key in enumerate(self._labels[category])}
            for category in self._keys
        }


def _data_from_dataframe(df):
    """
    Encodes a Pandas DataFrame as a NumPy ndarray
    """
    data = np.empty(df.shape, dtype=int)
    labels = OrderedDict()
    # encode each column
    for i, col in enumerate(df.columns):
        # get unique values, and encode data column by unique value indices
        unique_vals, encoding = np.unique(
            df[col].to_numpy(dtype=str), return_inverse=True
        )
        data[:, i] = encoding

        # add unique values to label dict
        labels.update({col: unique_vals})
    return data, labels
