from collections.abc import Hashable
import pandas as pd
import numpy as np

from hypernetx.classes.entity import Entity
from hypernetx.classes.helpers import update_properties


class EntitySet(Entity):
    """ Class for handling 2-dimensional (i.e., system of sets, bipartite) data when
    building network-like models, i.e., :class:`Hypergraph`

    Parameters
    ----------
    entity : `Entity`, `pandas.DataFrame`, dict of lists or sets, list of lists or sets, optional
        If an `Entity` with N levels or a `pandas.DataFrame` with N columns, represents
        N-dimensional entity data (data table);
        if N > 2, only considers levels (columns) `level1` and `level2`.
        Otherwise, represents 2-dimensional entity data (system of sets).
    data : `numpy.ndarray`, optional
        2D M x N ndarray of ints (data table)
        Sparse representation of an N-dimensional incidence tensor with M nonzero cells;
        if N > 2, only considers levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided.
    static : bool, default=True
        If True, entity data may not be altered, and the `state_dict` will never be cleared
        Otherwise, rows may be added to and removed from the data table, and updates will clear the `state_dict`
    labels : `collections.OrderedDict` of lists, optional
        User-specified labels in corresponding order to ints in `data`.
        For M x N `data`, N > 2, `labels` must contain either 2 or N keys; if N keys,
        only considers labels for levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided or `data` is not provided.
    uid : hashable, optional
        A unique identifier for the `StaticEntity`
    level1, level2 : int, default=0,1
        Each item in `level1` defines a set containing all the `level2` items with which
        it appears in the same row of the underlying data table.
        Ignored if `entity`, `data` (if `entity` not provided), and `labels` all (if
        provided) represent 1- or 2-dimensional data (set or system of sets)
    weights : array-like or hashable, optional
        User-specified cell weights corresponding to entity data;
        If array-like and `entity` or `data` defines a data table, length must equal the number of rows;
        If array-like and `entity` defines a system of sets, length must equal the total sum of the sizes of all sets;
        If hashable and `entity` is a `pandas.DataFrame`, must be the name of a column in `entity`;
        Otherwise, weight for all cells is assumed to be 1.
        Ignored if `entity` is an `Entity` and `keep_weights`=True
    keep_weights : bool, default=True
        If `entity` is an `Entity`, whether to preserve the existing cell weights or not;
        otherwise, ignored.
    aggregateby : {'last', count', 'sum', 'mean','median', max', 'min', 'first', 'last', None}, default='sum'
        Name of function to use for aggregating cell weights of duplicate rows when
        `entity` or `data` defines a data table.
        If None, duplicate rows will be dropped without aggregating cell weights.
        Effectively ignored if `entity` defines a system of sets
    properties : dict of dicts
        Nested dict of {item label: dict of {property name : property value}}
        User-specified properties to be assigned to individual items in the data,
            i.e., cell entries in a data table; sets or set elements in a system of sets
    cell_properties : dict of dicts of dicts
        Nested dict of {level1 item label: {level2 item label: dict of {cell property name : cell property value}}}
        User-specified properties to be assigned to cells of the incidence matrix,
            i.e., rows in a data table; pairs of (set, element of set) in a system of sets
    """
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

        if isinstance(entity, Entity):
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
            properties=properties,
        )

        self._cell_properties = (
            self._create_cell_properties(cell_properties)
            if self._dimsize == 2
            else None
        )

    @property
    def cell_properties(self):
        # Dev Note:
        return self._cell_properties

    @property
    def memberships(self):
        """
        Reverses the elements dictionary

        Returns
        -------
        dict
            Same as elements_by_level with level1 = 1, level2 = 0.
        """
        if self._dimsize == 1:
            return self._state_dict.get("memberships")

        return super().memberships

    def restrict_to_levels(
        self, levels, weights=False, aggregateby="sum", keep_memberships=True, **kwargs
    ):
        """
        Limit Static Entity data to specific levels

        Parameters
        ----------
        levels : array
            index of labels in data
        weights : bool, optional, default : False
            Whether or not to aggregate existing weights in self when
            restricting to levels. If False then weights will be assigned 1.
        aggregateby : str, optional, {None, 'last', count', 'sum', 'mean',
            'median', max', 'min', 'first', 'last'}, default : 'count' Method
            to aggregate cell_weights of duplicate rows in setsystem of type
            pandas.DataFrame. If None then all cell_weights will be set to 1.
        uid : None, optional

        Returns
        -------
        Static Entity class
            hnx.classes.entity.Entity
        """
        restricted = super().restrict_to_levels(levels, weights, aggregateby, **kwargs)

        if keep_memberships:
            restricted._state_dict["memberships"] = self.memberships

        return restricted

    def restrict_to(self, indices, **kwargs):
        """
        Limit Static Entityset data to specific indices of keys

        Parameters
        ----------
        indices : array
            array of indices in keys
        uid : None, optional

        Returns
        -------
        EntitySet
            hnx.classes.entity.EntitySet

        """
        return self.restrict_to_indices(indices, **kwargs)

    def _create_cell_properties(self, props):
        # Dev Note:
        index = pd.MultiIndex(levels=([], []), codes=([], []), names=self._data_cols)
        kwargs = {"index": index, "name": "cell_properties"}
        if props:
            cells = [(edge, node) for edge in props for node in props[edge]]
            index = pd.MultiIndex.from_tuples(cells, names=self._data_cols)
            data = [props[edge][node] for edge, node in index]
            kwargs.update(index=index, data=data)
        return pd.Series(**kwargs)

    def assign_cell_properties(self, props):
        # Dev Note: Not sure on this one
        if self._dimsize == 2:
            cell_properties = self._create_cell_properties(props)

            if not self._cell_properties.empty:
                cell_properties = update_properties(
                    self._cell_properties, cell_properties
                )

            self._cell_properties = cell_properties

    def collapse_identical_elements(self, return_equivalence_classes=False, **kwargs):
        """
        Returns EntitySet after collapsing elements if they have same
        children If no elements share same children, a copy of the original
        EntitySet is returned

        Parameters
        ----------
        uid : None, optional
        return_equivalence_classes : bool, optional
            If True, return a dictionary of equivalence classes keyed by new
            edge names


        Returns
        -------
        EntitySet
            hnx.classes.Entity.EntitySet
        """
        collapse = (
            self._dataframe[self._data_cols]
            .groupby(self._data_cols[0], as_index=False)
            .agg(frozenset)
        )
        agg_kwargs = {"name": (self._data_cols[0], lambda x: f"{x.iloc[0]}: {len(x)}")}
        if return_equivalence_classes:
            agg_kwargs.update(equivalence_class=(0, list))
        collapse = collapse.groupby(self._data_cols[1], as_index=False).agg(
            **agg_kwargs
        )
        collapse = collapse.set_index("name")
        new_entity_dict = collapse[self._data_cols[1]].to_dict()
        new_entity = EntitySet(new_entity_dict, **kwargs)
        if return_equivalence_classes:
            equivalence_classes = collapse.equivalence_class.to_dict()
            return new_entity, equivalence_classes
        return new_entity
