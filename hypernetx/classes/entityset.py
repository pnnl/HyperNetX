from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from typing import Optional, Any

import pandas as pd
import numpy as np
from ast import literal_eval

from hypernetx.classes import Entity
from hypernetx.classes.helpers import AttrList


class EntitySet(Entity):
    """Class for handling 2-dimensional (i.e., system of sets, bipartite) data when
    building network-like models, i.e., :class:`Hypergraph`

    Parameters
    ----------
    entity : Entity, pandas.DataFrame, dict of lists or sets, or list of lists or sets, optional
        If an ``Entity`` with N levels or a ``DataFrame`` with N columns,
        represents N-dimensional entity data (data table).
        If N > 2, only considers levels (columns) `level1` and `level2`.
        Otherwise, represents 2-dimensional entity data (system of sets).
    data : numpy.ndarray, optional
        2D M x N ``ndarray`` of ints (data table);
        sparse representation of an N-dimensional incidence tensor with M nonzero cells.
        If N > 2, only considers levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided.
    static : bool, default=True
        If True, entity data may not be altered,
        and the :attr:`state_dict <_state_dict>` will never be cleared.
        Otherwise, rows may be added to and removed from the data table,
        and updates will clear the :attr:`state_dict <_state_dict>`.
    labels : collections.OrderedDict of lists, optional
        User-specified labels in corresponding order to ``ints`` in `data`.
        For M x N `data`, N > 2, `labels` must contain either 2 or N keys.
        If N keys, only considers labels for levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided or `data` is not provided.
    uid : hashable, optional
        A unique identifier for the ``EntitySet``.
    level1, level2 : int, default=0,1
        Each item in `level1` defines a set containing all the `level2` items with which
        it appears in the same row of the underlying data table.
        Ignored if `entity`, `data` (if `entity` not provided), and `labels` all (if
        provided) represent 1- or 2-dimensional data (set or system of sets).
    weights : array_like or hashable, optional
        User-specified cell weights corresponding to entity data.
        If ``array_like`` and `entity` or `data` defines a data table,
            length must equal the number of rows.
        If ``array_like`` and `entity` defines a system of sets,
            length must equal the total sum of the sizes of all sets.
        If ``hashable`` and `entity` is a ``DataFrame``,
            must be the name of a column in `entity`.
        Otherwise, weight for all cells is assumed to be 1.
        Ignored if `entity` is an :class:`Entity` and `keep_weights`=True.
    keep_weights : bool, default=True
        Whether to preserve any existing cell weights.
        Ignored if `entity` is not an :class:`Entity`.
    aggregateby : {'sum', 'last', count', 'mean','median', max', 'min', 'first', None}, optional
        Name of function to use for aggregating cell weights of duplicate rows when
        `entity` or `data` defines a data table, default is "sum".
        If None, duplicate rows will be dropped without aggregating cell weights.
        Effectively ignored if `entity` defines a system of sets
    properties : dict of dicts
        Nested dict of ``{item label: dict of {property name: property value}}``.
        User-specified properties to be assigned to individual items in the data,
        i.e., cell entries in a data table; sets or set elements in a system of sets
    cell_properties : dict of dicts of dicts
        User-specified properties to be assigned to cells of the incidence matrix,
        i.e., rows in a data table; pairs of (set, element of set) in a system of sets.
        Ignored if underlying data is 1-dimensional (set).
        If Hashable and `entity` is an :class:`Entity` or ``DataFrame``, name of a
        column in the data table containing entries of the form
        ``{cell property name: cell property value}``.
        If ``DataFrame``, each row gives
        ``[level1 item, level2 item, {cell property name: cell property value}]``
        (order of columns does not matter; column names for `level1` and `level2`
        must be the same as in :attr:`dataframe`).
        If doubly-nested dict,
        ``{level1 item: {level2 item: {cell property name: cell property value}}}``.
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
        level1: str | int = 0,
        level2: str | int = 1,
        weights: Optional[Sequence | Hashable] = None,
        keep_weights: bool = True,
        aggregateby: str = "sum",
        properties: Optional[dict[str, dict[str, Any]]] = None,
        cell_properties: Optional[
            Hashable | pd.DataFrame | dict[str, dict[str, dict[str, Any]]]
        ] = None,
        cell_props_col="cell_properties",
        **kwargs,
    ):
        self._cell_props_col = cell_props_col

        # if the entity data is passed as an Entity, get its underlying data table and
        # proceed to the case for entity data passed as a DataFrame
        if isinstance(entity, Entity):
            if keep_weights:
                # preserve original weights
                weights = entity._cell_weight_col
            entity = entity.dataframe

        # if the entity data is passed as a DataFrame, restrict to two columns if needed
        if isinstance(entity, pd.DataFrame) and len(entity.columns) > 2:
            # metadata columns are not considered levels of data,
            # remove them before indexing by level

            if isinstance(cell_properties, Hashable):
                cell_properties = [cell_properties]

            prop_cols = []
            if isinstance(cell_properties, list):
                for col in cell_properties:
                    try:
                        if col in entity:
                            prop_cols.append(col)
                    except TypeError:
                        continue

            try:
                weight_col = [weights] if weights in entity else []
            except TypeError:
                weight_col = []

            meta_cols = prop_cols + weight_col

            # if both levels are column names, no need to index by level
            if isinstance(level1, str) and isinstance(level2, str):
                columns = [level1, level2]
            # if one or both of the levels are given by index, get column name
            else:
                all_columns = entity.columns.drop(meta_cols)
                columns = [
                    all_columns[lev] if isinstance(lev, int) else lev
                    for lev in (level1, level2)
                ]

            # if there is a column for cell properties, convert to separate DataFrame
            if prop_cols:
                cell_properties = entity[[*columns, *prop_cols]]
                self._cell_props_col = prop_cols[0]
            # if there is a column for weights, preserve it
            columns += weight_col

            # pass level1, level2, and weights (optional) to Entity constructor
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
            **kwargs,
        )

        # if underlying data is 2D (system of sets), create and assign cell properties
        self._cell_properties = (
            self._create_cell_properties(cell_properties)
            if self._dimsize == 2
            else None
        )

    @property
    def cell_properties(self) -> Optional[pd.DataFrame]:
        """Properties assigned to cells of the incidence matrix

        Returns
        -------
        pandas.Series, optional
            Returns None if :attr:`dimsize`=1
        """
        return self._cell_properties

    @property
    def memberships(self) -> dict[str, AttrList[str]]:
        """Extends :attr:`Entity.memberships`

        Each item in level 1 (second column) defines a set containing all the level 0
        (first column) items with which it appears in the same row of the underlying
        data table.

        Returns
        -------
        dict of AttrList
            System of sets representation as dict of
            ``{level 1 item: AttrList(level 0 items)}``.

        See Also
        --------
        elements : dual of this representation,
                   i.e., each item in level 0 (first column) defines a set
        restrict_to_levels : for more information on how memberships work for
                             1-dimensional (set) data
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
        """Extends :meth:`Entity.restrict_to_levels`

        Parameters
        ----------
        levels : array_like of int
            indices of a subset of levels (columns) of data
        weights : bool, default=False
            If True, aggregate existing cell weights to get new cell weights.
            Otherwise, all new cell weights will be 1.
        aggregateby : {'sum', 'last', count', 'mean','median', max', 'min', 'first', None}, optional
            Method to aggregate weights of duplicate rows in data table;
            If None or `weights`=False then all new cell weights will be 1
        keep_memberships : bool, default=True
            Whether to preserve membership information for the discarded level when
            the new ``EntitySet`` is restricted to a single level
        **kwargs
            Extra arguments to :class:`EntitySet` constructor

        Returns
        -------
        EntitySet
        """
        restricted = super().restrict_to_levels(
            levels, weights, aggregateby, cell_props_col=self._cell_props_col, **kwargs
        )

        if keep_memberships:
            # use original memberships to set memberships for the new EntitySet
            # TODO: This assumes levels=[1], add explicit checks for other cases
            restricted._state_dict["memberships"] = self.memberships

        return restricted

    def restrict_to(self, indices: int | Iterable[int], **kwargs) -> EntitySet:
        """Alias of :meth:`restrict_to_indices` with default parameter `level`=0

        Parameters
        ----------
        indices : array_like of int
            indices of item label(s) in `level` to restrict to
        **kwargs
            Extra arguments to :class:`EntitySet` constructor

        Returns
        -------
        EntitySet

        See Also
        --------
        restrict_to_indices
        """
        restricted = self.restrict_to_indices(
            indices, cell_props_col=self._cell_props_col, **kwargs
        )
        if not self.cell_properties.empty:
            cell_properties = self.cell_properties.loc[
                list(restricted.uidset)
            ].reset_index()
            restricted.assign_cell_properties(cell_properties)
        return restricted

    def _create_cell_properties(
        self,
        props: Optional[pd.DataFrame | dict[str, dict[str, dict[str, Any]]]] = None,
    ) -> pd.DataFrame:
        """Helper function for :meth:`assign_cell_properties`

        Parameters
        ----------
        props : pandas.DataFrame or doubly-nested dict, optional
            If ``DataFrame``, each row gives
            ``[level 0 item, level 1 item, {cell property name: cell property value}]``
            (order of columns does not matter;
            column names for level 0 and 1 must be the same as in :attr:`dataframe`).
            Otherwise, doubly-nested dict of
            ``{level 0 item: {level 1 item: {cell property name: cell property value}}}``

        Returns
        -------
        pandas.DataFrame
            with ``MultiIndex`` on ``(level 0 item, level 1 item)``;
            each entry holds dict of ``{cell property name: cell property value}``
        """

        if isinstance(props, pd.DataFrame):
            # TODO: assumes there are no row duplicates,
            #       add checks to handle removal if there are
            index = None
            # set MultiIndex on (level 0, level1)
            data = props.set_index(self._data_cols)
            if self._cell_props_col not in data:
                data[self._cell_props_col] = [{} for _ in range(len(data))]
            elif isinstance(data[self._cell_props_col].iloc[0], str):
                data[self._cell_props_col] = data[self._cell_props_col].apply(
                    literal_eval
                )

        elif isinstance(props, dict):
            # construct MultiIndex from all (level 0 item, level 1 item) pairs from
            # nested keys of props dict
            cells = [(edge, node) for edge in props for node in props[edge]]
            index = pd.MultiIndex.from_tuples(cells, names=self._data_cols)
            # properties for each cell
            data = [props[edge][node] for edge, node in index]

        else:
            # hierarchical index over columns of data
            index = pd.MultiIndex(
                levels=([], []), codes=([], []), names=self._data_cols
            )
            # empty if no valid props provided
            data = None

        return pd.DataFrame(data=data, index=index).sort_index()

    def assign_cell_properties(
        self,
        props: pd.DataFrame | dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        """Assign new properties to cells of the incidence matrix and update
        :attr:`properties`

        Parameters
        ----------
        props : pandas.DataFrame or doubly-nested dict
            If ``DataFrame``, each row gives
            ``[level 0 item, level 1 item, {cell property name: cell property value}]``
            (order of columns does not matter;
            column names for level 0 and 1 must be the same as in :attr:`dataframe`).
            Otherwise, doubly-nested dict of
            ``{level 0 item: {level 1 item: {cell property name: cell property value}}}``

        Notes
        -----
        Not supported for :attr:`dimsize`=1
        """
        if self._dimsize == 2:
            # convert nested dict of cell properties to MultiIndexed Series
            cell_properties = self._create_cell_properties(props)

            # update stored cell properties
            self._cell_properties = self._cell_properties.combine_first(cell_properties)
            self._cell_properties.update(cell_properties)

    def collapse_identical_elements(
        self, return_equivalence_classes: bool = False, **kwargs
    ) -> EntitySet | tuple[EntitySet, dict[str, list[str]]]:
        """Create a new :class:`EntitySet` by collapsing sets with the same set elements

        Each item in level 0 (first column) defines a set containing all the level 1
        (second column) items with which it appears in the same row of the underlying
        data table.

        Parameters
        ----------
        return_equivalence_classes : bool, default=False
            If True, return a dictionary of equivalence classes keyed by new edge names
        **kwargs
            Extra arguments to :class:`EntitySet` constructor

        Returns
        -------
        new_entity : EntitySet
            new :class:`EntitySet` with identical sets collapsed;
            if all sets are unique, the system of sets will be the same as the original.
        equivalence_classes : dict of lists, optional
            if `return_equivalence_classes`=True,
            ``{collapsed set label: [level 0 item labels]}``.
        """
        # group by level 0 (set), aggregate level 1 (set elements) as frozenset
        collapse = (
            self._dataframe[self._data_cols]
            .groupby(self._data_cols[0], as_index=False)
            .agg(frozenset)
        )

        # aggregation method to rename equivalence classes as [first item]: [# items]
        agg_kwargs = {"name": (self._data_cols[0], lambda x: f"{x.iloc[0]}: {len(x)}")}
        if return_equivalence_classes:
            # aggregation method to list all items in each equivalence class
            agg_kwargs.update(equivalence_class=(self._data_cols[0], list))
        # group by frozenset of level 1 items (set elements), aggregate to get names of
        # equivalence classes and (optionally) list of level 0 items (sets) in each
        collapse = collapse.groupby(self._data_cols[1], as_index=False).agg(
            **agg_kwargs
        )
        # convert to nested dict representation of collapsed system of sets
        collapse = collapse.set_index("name")
        new_entity_dict = collapse[self._data_cols[1]].to_dict()
        # construct new EntitySet from system of sets
        new_entity = EntitySet(new_entity_dict, **kwargs)

        if return_equivalence_classes:
            # lists of equivalent sets, keyed by equivalence class name
            equivalence_classes = collapse.equivalence_class.to_dict()
            return new_entity, equivalence_classes
        return new_entity

    def set_cell_property(self, item1, item2, prop_name, prop_val):
        if (item1, item2) in self.cell_properties.index:
            if prop_name in self.properties:
                self._cell_properties.loc[(item1, item2), prop_name] = pd.Series(
                    [prop_val]
                )
            else:
                self._cell_properties.loc[(item1, item2), self._cell_props_col].update(
                    {prop_name: prop_val}
                )
