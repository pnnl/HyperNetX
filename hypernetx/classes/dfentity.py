from hypernetx import *
import pandas as pd
import numpy as np
from collections import OrderedDict


class StaticEntity(object):
    def __init__(
        self,
        data,  # DataFrame, Dict of Lists, List of Lists, or np array
        labels=None,
        weights=None,  # array-like of values corresponding to rows of data
        aggregateby="sum",
    ):
        if isinstance(data, pd.DataFrame):
            # dataframe case
            self._dataframe = data
        elif isinstance(data, dict):
            # dict of lists case
            k = sum([[i] * len(data[i]) for i in data], [])
            v = sum(data.values(), [])
            self._dataframe = pd.DataFrame({0: k, 1: v})
        elif isinstance(data, list):
            # list of lists case
            k = sum([[i] * len(data[i]) for i in range(len(data))], [])
            v = sum(data, [])
            self._dataframe = pd.DataFrame({0: k, 1: v})
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self._dataframe = pd.DataFrame(data)

        self._data_cols = self._dataframe.columns

        if isinstance(weights, (list, np.ndarray)) and len(weights) == len(
            self._dataframe
        ):
            self._dataframe["cell_weights"] = weights
            self._cell_weight_cols = ["cell_weights"]
        elif weights in self._dataframe:
            self._cell_weight_cols = [weights]
        else:
            self._dataframe["cell_weights"] = np.ones(len(self._dataframe), dtype=int)
            self._cell_weight_cols = ["cell_weights"]

        self._data_cols = list(self._dataframe.columns.drop(self._cell_weight_cols))

        if aggregateby is None:
            self._dataframe.drop_duplicates(subset=self._data_cols, inplace=True)
        else:
            self._dataframe = self._dataframe.groupby(
                self._data_cols, as_index=False, sort=False
            ).agg({w: aggregateby for w in self._cell_weight_cols})

        self._dataframe[self._data_cols] = self._dataframe[self._data_cols].astype(
            "category"
        )
        # self._dimensions = tuple(self._data[self._data_cols].nunique())
        self._dimsize = len(self._data_cols)

        self._state_dict = {}

        if isinstance(labels, dict):
            self._state_dict['labels'] = labels

    @property
    def data(self):
        if "data" not in self._state_dict:
            # assumes dtype of data cols is categorical (dataframe not altered since init)
            self._state_dict["data"] = (
                self._dataframe[self._data_cols].apply(lambda x: x.cat.codes).to_numpy()
            )
            self._state_dict["labels"] = (
                self._dataframe[self._data_cols]
                .apply(lambda x: x.cat.categories.to_list())
                .to_dict()
            )

        return self._state_dict["data"]

    @property
    def cell_weights(self):
        if "cell_weights" not in self._state_dict:
            self._state_dict["cell_weights"] = self._dataframe.set_index(
                self._data_cols
            )[self._cell_weight_cols].to_dict()

        return self._state_dict["cell_weights"]

    @property
    def dimensions(self):
        if "dimensions" not in self._state_dict:
            self._state_dict["dimensions"] = tuple(
                self._dataframe[self._data_cols].nunique()
            )

        return self._state_dict["dimensions"]

    @property
    def dimsize(self):
        return self._dimsize

    @property
    def uidset(self):
        return self.uidset_by_level(0)

    @property
    def children(self):
        return self.uidset_by_level(1)

    def uidset_by_level(self, level):
        data = self._dataframe[self._data_cols]
        col = data.columns[level]
        return self.uidset_by_column(col)

    def uidset_by_column(self, column):
        if "uidset" not in self._state_dict:
            self._state_dict["uidset"] = {}
        if column not in self._state_dict["uidset"]:
            self._state_dict["uidset"][column] = set(
                self._dataframe[column].dropna().unique()
            )

        return self._state_dict["uidset"][column]

    @property
    def elements(self):
        return self.elements_by_level(0, 1)

    @property
    def incidence_dict(self):
        return self.elements

    @property
    def memberships(self):
        return self.elements_by_level(1, 0)

    def elements_by_level(self, level1, level2):
        data = self._dataframe[self._data_cols]
        col1, col2 = data.columns[[level1, level2]]
        return self.elements_by_column(col1, col2)

    def elements_by_column(self, col1, col2):
        if "elements" not in self._state_dict:
            self._state_dict["elements"] = defaultdict(dict)
        if (
            col1 not in self._state_dict["elements"]
            or col2 not in self._state_dict["elements"][col1]
        ):
            self._state_dict["elements"][col1][col2] = (
                self._dataframe.groupby(col1)[col2].apply(list).to_dict()
            )

        return self._state_dict["elements"][col1][col2]

    @property
    def dataframe(self):
        return self._dataframe

    def size(self, level=0):
        return self.dimensions[level]

    def is_empty(self, level=0):
        return self.size(level) == 0

    def __len__(self):
        return self.dimensions[0]

    def __contains__(self):
        # Need to define labels
        return item in np.concatenate(list(self._labels.values()))
    
    def __getitem__(self, item):
        return self.elements[item]

    def __iter__(self):
        return iter(self.elements)
    
    def __call__(self, label_index=0):
        # Need to define labels
        return iter(self._labs[label_index])