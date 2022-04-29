import pandas as pd
from hypernetx.classes.staticentity import StaticEntity
import warnings
from hypernetx import *
from pandas.api.types import CategoricalDtype
import numpy as np
from collections import defaultdict, OrderedDict, UserList
from collections.abc import Hashable
from scipy.sparse import csr_matrix
from hypernetx.classes.helpers import *


class StaticEntitySet(StaticEntity):
    def __init__(
        self,
        entity=None,
        data=None,
        static=True,
        labels=None,
        uid=None,
        level1=0,
        level2=1,
        weights=None,
        keep_weights=True,
        aggregateby="sum",
        properties=None,
        cell_properties=None,
    ):

        if isinstance(entity, StaticEntity):
            if keep_weights:
                weights = entity._cell_weight_col
            entity = entity.dataframe

        if isinstance(entity, pd.DataFrame) and len(entity.columns) > 2:
            if isinstance(weights, Hashable) and weights in entity:
                columns = entity.columns.drop(weights)[[level1, level2]]
                columns = columns.append(pd.Index([weights]))
            else:
                columns = entity.columns[[level1, level2]]
            entity = entity[columns]

        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] > 2:
            data = data[:, (level1, level2)]

        if isinstance(labels, dict) and len(labels) > 2:
            label_keys = list(labels)
            columns = (label_keys[level1], label_keys[level2])
            labels = {col: labels[col] for col in columns}

        super().__init__(
            entity=entity,
            data=data,
            static=static,
            labels=labels,
            uid=uid,
            weights=weights,
            aggregateby=aggregateby,
            properties=properties
        )

        self._cell_properties = self._create_cell_properties(cell_properties) if self._dimsize == 2 else None

    @property
    def cell_properties(self):
        return self._cell_properties

    @property
    def memberships(self):
        if self._dimsize == 1:
            return self._state_dict.get("memberships")

        return super().memberships

    def restrict_to_levels(
        self, levels, weights=False, aggregateby="sum", keep_memberships=True, **kwargs
    ):
        restricted = super().restrict_to_levels(levels, weights, aggregateby, **kwargs)

        if keep_memberships:
            restricted._state_dict["memberships"] = self.memberships

        return restricted

    def restrict_to(self, indices, **kwargs):
        return self.restrict_to_indices(indices, **kwargs)

    def _create_cell_properties(self, props):
        index = pd.MultiIndex(levels=([],[]),codes=([],[]),names=self._data_cols)
        kwargs = {'index':index,'name':'cell_properties'}
        if props:
            cells = [(edge,node) for edge in props for node in props[edge]]
            index = pd.MultiIndex.from_tuples(cells, names=self._data_cols)
            data = [props[edge][node] for edge, node in index]
            kwargs.update(index=index,data=data)
        return pd.Series(**kwargs)

    def assign_cell_properties(self, props):
        if self._dimsize == 2:
            cell_properties = self._create_cell_properties(props)

            if not self._cell_properties.empty:
                cell_properties = update_properties(self._cell_properties, cell_properties)

            self._cell_properties = cell_properties

