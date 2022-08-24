from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from typing import Optional, Any

import pandas as pd
import numpy as np

from hypernetx.classes.entity import Entity
from hypernetx.classes.helpers import update_properties, AttrList


class EntitySet(Entity):
    """Class for handling 2-dimensional (i.e., system of sets, bipartite) data when
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
        Ignored if underlying data is 1-dimensional (set)
    """

    def __init__(
        self,
        entity: Optional[
            Entity | pd.DataFrame | dict[Iterable] | list[Iterable]
        ] = None,
        data: Optional[np.ndarray] = None,
        static: bool = True,
        labels: Optional[OrderedDict[str, list[str]]] = None,
        uid: Optional[Hashable] = None,
        level1: int = 0,
        level2: int = 1,
        weights: Optional[Sequence | Hashable] = None,
        keep_weights: bool = True,
        aggregateby: str = "sum",
        properties: Optional[dict[str, dict[str, Any]]] = None,
        cell_properties: Optional[dict[str, dict[str, dict[str, Any]]]] = None,
    ):
        # if the entity data is passed as an Entity, get its underlying data table and
        # proceed to the case for entity data passed as a DataFrame
        if isinstance(entity, Entity):
            if keep_weights:
                # preserve original weights
                weights = entity._cell_weight_col
            entity = entity.dataframe

        # if the entity data is passed as a DataFrame, restrict to two columns if needed
        if isinstance(entity, pd.DataFrame) and len(entity.columns) > 2:
            # if there is a column for weights, preserve it
            if isinstance(weights, Hashable) and weights in entity:
                columns = entity.columns.drop(weights)[[level1, level2]]
                columns = columns.append(pd.Index([weights]))
            else:
                columns = entity.columns[[level1, level2]]
            entity = entity[columns]

        # if a 2D ndarray is passed, restrict to two columns if needed
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] > 2:
            data = data[:, (level1, level2)]

        # if a dict of labels is provided, restrict to labels for two columns if needed
        if isinstance(labels, dict) and len(labels) > 2:
            label_keys = list(labels)
            columns = (label_keys[level1], label_keys[level2])
            labels = {col: labels[col] for col in columns}

        # pass reformatted params to Entity constructor
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

        # if underlying data is 2D (system of sets), create and assign cell properties
        self._cell_properties = (
            self._create_cell_properties(cell_properties)
            if self._dimsize == 2
            else None
        )

    @property
    def cell_properties(self) -> pd.Series:
        """Properties assigned to cells of the incidence matrix

        Returns
        -------
        pandas.Series, optional
            Returns None if dimsize=1
        """
        return self._cell_properties

    @property
    def memberships(self) -> dict[str, AttrList[str]]:
        """Extends Entity.memberships

        Each item in level 1 (second column) defines a set containing all the level 0
        (first column) items with which it appears in the same row of the underlying
        data table.

        Returns
        -------
        dict of `AttrList`
            System of sets representation as dict of {level 1 item : AttrList(level 0 items)}

        See Also
        --------
        elements : dual of this representation i.e., each item in level 0 (first column) defines a set
        restrict_to_levels : for more information on how memberships work for 1-dimensional (set) data
        """
        if self._dimsize == 1:
            return self._state_dict.get("memberships")

        return super().memberships

    def restrict_to_levels(
        self,
        levels: int | Iterable[int],
        weights: bool = False,
        aggregateby: Optional[str] = "sum",
        keep_memberships: bool = True,
        **kwargs,
    ) -> EntitySet:
        """Extends Entity.restrict_to_levels

        Parameters
        ----------
        levels : array_like of int
            indices of a subset of levels (columns) of data
        weights : bool, default=False
            If True, aggregate existing cell weights to get new cell weights
            Otherwise, all new cell weights will be 1
        aggregateby : {'last', count', 'sum', 'mean','median', max', 'min', 'first', None}, default='sum'
            Method to aggregate weights of duplicate rows in data table
            If None or weights=False then all new cell weights will be 1
        keep_memberships : bool, default=True
            Whether to preserve membership information for the discarded level when
            the new EntitySet is restricted to a single level
        **kwargs
            Extra arguments to `StaticEntity` constructor

        Returns
        -------
        EntitySet
        """
        restricted = super().restrict_to_levels(levels, weights, aggregateby, **kwargs)

        if keep_memberships:
            # use original memberships to set memberships for the new EntitySet
            # TODO: This assumes levels=[1], add explicit checks for other cases
            restricted._state_dict["memberships"] = self.memberships

        return restricted

    def restrict_to(self, indices: int | Iterable[int], **kwargs) -> EntitySet:
        """Alias of `restrict_to_indices` with default parameter `level`=0

        Parameters
        ----------
        indices : array_like of int
            indices of item label(s) in `level` to restrict to
        **kwargs
            Extra arguments to `StaticEntity` constructor

        Returns
        -------
        EntitySet

        See Also
        --------
        restrict_to_indices
        """
        return self.restrict_to_indices(indices, **kwargs)

    def _create_cell_properties(
        self, props: dict[str, dict[str, dict[str, Any]]]
    ) -> pd.Series:
        """Helper function for `assign_cell_properties`

        Parameters
        ----------
        props : dict of dicts of dicts
            Nested dict of {level 0 item label: dict of {level 1 item label: dict of {cell property name : cell property value}}}

        Returns
        -------
        pandas.Series
            MultiIndex of (level 0 item, level 1 item), each entry holds dict of {cell property name: cell property value}
        """
        index = pd.MultiIndex(levels=([], []), codes=([], []), names=self._data_cols)
        kwargs = {"index": index, "name": "cell_properties"}
        if props:
            cells = [(edge, node) for edge in props for node in props[edge]]
            index = pd.MultiIndex.from_tuples(cells, names=self._data_cols)
            data = [props[edge][node] for edge, node in index]
            kwargs.update(index=index, data=data)
        return pd.Series(**kwargs)

    def assign_cell_properties(
        self, props: dict[str, dict[str, dict[str, Any]]]
    ) -> None:
        """Assign new properties to cells of the incidence matrix and update `self.properties`

        Parameters
        ----------
        props : dict of dicts of dicts
            Nested dict of {level 0 item label: dict of {level 1 item label: dict of {cell property name : cell property value}}}

        Notes
        -----
        Not supported for dimsize=1
        """
        if self._dimsize == 2:
            cell_properties = self._create_cell_properties(props)

            if not self._cell_properties.empty:
                cell_properties = update_properties(
                    self._cell_properties, cell_properties
                )

            self._cell_properties = cell_properties

    def collapse_identical_elements(
        self, return_equivalence_classes: bool = False, **kwargs
    ) -> EntitySet | tuple[EntitySet, dict[str, list[str]]]:
        """Create a new EntitySet by collapsing sets with the same set elements

        Each item in level 0 (first column) defines a set containing all the level 1
        (second column) items with which it appears in the same row of the underlying
        data table.

        Parameters
        ----------
        return_equivalence_classes : bool, default=False
            If True, return a dictionary of equivalence classes keyed by new edge names
        **kwargs
            Extra arguments to `StaticEntity` constructor

        Returns
        -------
        new_entity : EntitySet
            new EntitySet with identical sets collapsed; if all sets are unique, the set
            system will be the same as the original
        equivalence_classes : dict of lists, optional
            if `return_equivalence_classes`=True, {collapsed set label: [level 0 item labels]}
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
