from __future__ import annotations

import warnings
from ast import literal_eval
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Mapping
from typing import Optional, Any, TypeVar, Union
from pprint import pformat

import numpy as np
import pandas as pd

from hypernetx.classes import Entity
from hypernetx.classes.helpers import AttrList

# from hypernetx.utils.log import get_logger

# _log = get_logger("entity_set")

T = TypeVar("T", bound=Union[str, int])


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
        2D M x N ``ndarray`` of ``ints`` (data table);
        sparse representation of an N-dimensional incidence tensor with M nonzero cells.
        If N > 2, only considers levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided.
    labels : collections.OrderedDict of lists, optional
        User-specified labels in corresponding order to ``ints`` in `data`.
        For M x N `data`, N > 2, `labels` must contain either 2 or N keys.
        If N keys, only considers labels for levels (columns) `level1` and `level2`.
        Ignored if `entity` is provided or `data` is not provided.
    level1, level2 : str or int, default=0,1
        Each item in `level1` defines a set containing all the `level2` items with which
        it appears in the same row of the underlying data table.
        If ``int``, gives the index of a level;
        if ``str``, gives the name of a column in `entity`.
        Ignored if `entity`, `data` (if `entity` not provided), and `labels` all (if
        provided) represent 1- or 2-dimensional data (set or system of sets).
    weights : str or sequence of float, optional
        User-specified cell weights corresponding to entity data.
        If sequence of ``floats`` and `entity` or `data` defines a data table,
            length must equal the number of rows.
        If sequence of ``floats`` and `entity` defines a system of sets,
            length must equal the total sum of the sizes of all sets.
        If ``str`` and `entity` is a ``DataFrame``,
            must be the name of a column in `entity`.
        Otherwise, weight for all cells is assumed to be 1.
        Ignored if `entity` is an ``Entity`` and `keep_weights`=True.
    keep_weights : bool, default=True
        Whether to preserve any existing cell weights;
        ignored if `entity` is not an ``Entity``.
    cell_properties : str, list of str, pandas.DataFrame, or doubly-nested dict, optional
        User-specified properties to be assigned to cells of the incidence matrix, i.e.,
        rows in a data table; pairs of (set, element of set) in a system of sets.
        See Notes for detailed explanation.
        Ignored if underlying data is 1-dimensional (set).
        If doubly-nested dict,
        ``{level1 item: {level2 item: {cell property name: cell property value}}}``.
    misc_cell_props_col : str, default='cell_properties'
        Column name for miscellaneous cell properties; see Notes for explanation.
    kwargs
        Keyword arguments passed to the ``Entity`` constructor, e.g., `static`,
        `uid`, `aggregateby`, `properties`, etc. See :class:`Entity` for documentation
        of these parameters.

    Notes
    -----
    A **cell property** is a named attribute assigned jointly to a set and one of its
    elements, i.e, a cell of the incidence matrix.

    When an ``Entity`` or ``DataFrame`` is passed to the `entity` parameter of the
    constructor, it should represent a data table:

    +--------------+--------------+--------------+-------+--------------+
    | Column_1     | Column_2     | Column_3     | [...] | Column_N     |
    +==============+==============+==============+=======+==============+
    | level 1 item | level 2 item | level 3 item | ...   | level N item |
    +--------------+--------------+--------------+-------+--------------+
    | ...          | ...          | ...          | ...   | ...          |
    +--------------+--------------+--------------+-------+--------------+

    Assuming the default values for parameters `level1`, `level2`, the data table will
    be restricted to the set system defined by Column 1 and Column 2.
    Since each row of the data table represents an incidence or cell, values from other
    columns may contain data that should be converted to cell properties.

    By passing a **column name or list of column names** as `cell_properties`, each
    given column will be preserved in the :attr:`cell_properties` as an explicit cell
    property type. An additional column in :attr:`cell_properties` will be created to
    store a ``dict`` of miscellaneous cell properties, which will store cell properties
    of types that have not been explicitly defined and do not have a dedicated column
    (which may be assigned after construction). The name of the miscellaneous column is
    determined by `misc_cell_props_col`.

    You can also pass a **pre-constructed table** to `cell_properties` as a
    ``DataFrame``:

    +----------+----------+----------------------------+-------+-----------------------+
    | Column_1 | Column_2 | [explicit cell prop. type] | [...] | misc. cell properties |
    +==========+==========+============================+=======+=======================+
    | level 1  | level 2  | cell property value        | ...   | {cell property name:  |
    | item     | item     |                            |       | cell property value}  |
    +----------+----------+----------------------------+-------+-----------------------+
    | ...      | ...      | ...                        | ...   | ...                   |
    +----------+----------+----------------------------+-------+-----------------------+

    Column 1 and Column 2 must have the same names as the corresponding columns in the
    `entity` data table, and `misc_cell_props_col` can be used to specify the name of the
    column to be used for miscellaneous cell properties. If no column by that name is
    found, a new column will be created and populated with empty ``dicts``. All other
    columns will be considered explicit cell property types. The order of the columns
    does not matter.

    Both of these methods assume that there are no row duplicates in the tables passed
    to `entity` and/or `cell_properties`; if duplicates are found, all but the first
    occurrence will be dropped.

    """

    def __init__(
        self,
        entity: Optional[
            pd.DataFrame
            | np.ndarray
            | Mapping[T, Iterable[T]]
            | Iterable[Iterable[T]]
            | Mapping[T, Mapping[T, Mapping[T, Any]]]
        ] = None,
        data: Optional[np.ndarray] = None,
        labels: Optional[OrderedDict[T, Sequence[T]]] = None,
        level1: str | int = 0,
        level2: str | int = 1,
        weight_col: str | int = "cell_weights",
        weights: Sequence[float] | float | int | str = 1,
        # keep_weights: bool = True,
        cell_properties: Optional[
            Sequence[T] | pd.DataFrame | dict[T, dict[T, dict[Any, Any]]]
        ] = None,
        misc_cell_props_col: str = "cell_properties",
        uid: Optional[Hashable] = None,
        aggregateby: Optional[str] = "sum",
        properties: Optional[pd.DataFrame | dict[int, dict[T, dict[Any, Any]]]] = None,
        misc_props_col: str = "properties",
        # level_col: str = "level",
        # id_col: str = "id",
        **kwargs,
    ):
        self._misc_cell_props_col = misc_cell_props_col

        # if the entity data is passed as an Entity, get its underlying data table and
        # proceed to the case for entity data passed as a DataFrame
        # if isinstance(entity, Entity):
        #        #     _log.info(f"Changing entity from type {Entity} to {type(entity.dataframe)}")
        #     if keep_weights:
        #         # preserve original weights
        #         weights = entity._cell_weight_col
        #     entity = entity.dataframe

        # if the entity data is passed as a DataFrame, restrict to two columns if needed
        if isinstance(entity, pd.DataFrame) and len(entity.columns) > 2:
            #            _log.info(f"Processing parameter of 'entity' of type {type(entity)}...")
            # metadata columns are not considered levels of data,
            # remove them before indexing by level
            # if isinstance(cell_properties, str):
            #     cell_properties = [cell_properties]

            prop_cols = []
            if isinstance(cell_properties, Sequence):
                for col in {*cell_properties, self._misc_cell_props_col}:
                    if col in entity:
                        #                        _log.debug(f"Adding column to prop_cols: {col}")
                        prop_cols.append(col)

            # meta_cols = prop_cols
            # if weights in entity and weights not in meta_cols:
            #     meta_cols.append(weights)
            #            # _log.debug(f"meta_cols: {meta_cols}")
            if weight_col in prop_cols:
                prop_cols.remove(weight_col)
            if not weight_col in entity:
                entity[weight_col] = weights

            # if both levels are column names, no need to index by level
            if isinstance(level1, int):
                level1 = entity.columns[level1]
            if isinstance(level2, int):
                level2 = entity.columns[level2]
            # if isinstance(level1, str) and isinstance(level2, str):
            columns = [level1, level2, weight_col] + prop_cols
            # if one or both of the levels are given by index, get column name
            # else:
            #     all_columns = entity.columns.drop(meta_cols)
            #     columns = [
            #         all_columns[lev] if isinstance(lev, int) else lev
            #         for lev in (level1, level2)
            #     ]

            # if there is a column for cell properties, convert to separate DataFrame
            # if len(prop_cols) > 0:
            #     cell_properties = entity[[*columns, *prop_cols]]

            # if there is a column for weights, preserve it
            # if weights in entity and weights not in prop_cols:
            #     columns.append(weights)
            #            _log.debug(f"columns: {columns}")

            # pass level1, level2, and weights (optional) to Entity constructor
            entity = entity[columns]

        # if a 2D ndarray is passed, restrict to two columns if needed
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] > 2:
            #            _log.info(f"Processing parameter 'data' of type {type(data)}...")
            data = data[:, (level1, level2)]

        # if a dict of labels is provided, restrict to labels for two columns if needed
        if isinstance(labels, dict) and len(labels) > 2:
            label_keys = list(labels)
            columns = (label_keys[level1], label_keys[level2])
            labels = {col: labels[col] for col in columns}
        #            _log.debug(f"Restricted labels to columns:\n{pformat(labels)}")

        #        _log.info(
        #     f"Creating instance of {Entity} using reformatted params: \n\tentity: {type(entity)} \n\tdata: {type(data)} \n\tlabels: {type(labels)}, \n\tweights: {weights}, \n\tkwargs: {kwargs}"
        # )
        #        _log.debug(f"entity:\n{pformat(entity)}")
        #        _log.debug(f"data: {pformat(data)}")
        super().__init__(
            entity=entity,
            data=data,
            labels=labels,
            uid=uid,
            weight_col=weight_col,
            weights=weights,
            aggregateby=aggregateby,
            properties=properties,
            misc_props_col=misc_props_col,
            **kwargs,
        )

        # if underlying data is 2D (system of sets), create and assign cell properties
        if self.dimsize == 2:
            # self._cell_properties = pd.DataFrame(
            #     columns=[*self._data_cols, self._misc_cell_props_col]
            # )
            self._cell_properties = pd.DataFrame(self._dataframe)
            self._cell_properties.set_index(self._data_cols, inplace=True)
            if isinstance(cell_properties, (dict, pd.DataFrame)):
                self.assign_cell_properties(cell_properties)
        else:
            self._cell_properties = None

    @property
    def cell_properties(self) -> Optional[pd.DataFrame]:
        """Properties assigned to cells of the incidence matrix

        Returns
        -------
        pandas.Series, optional
            Returns None if :attr:`dimsize` < 2
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
        levels : array-like of int
            indices of a subset of levels (columns) of data
        weights : bool, default=False
            If True, aggregate existing cell weights to get new cell weights.
            Otherwise, all new cell weights will be 1.
        aggregateby : {'sum', 'first', 'last', 'count', 'mean', 'median', 'max', \
    'min', None}, optional
            Method to aggregate weights of duplicate rows in data table
            If None or `weights`=False then all new cell weights will be 1
        keep_memberships : bool, default=True
            Whether to preserve membership information for the discarded level when
            the new ``EntitySet`` is restricted to a single level
        **kwargs
            Extra arguments to :class:`EntitySet` constructor

        Returns
        -------
        EntitySet

        Raises
        ------
        KeyError
            If `levels` contains any invalid values
        """
        restricted = super().restrict_to_levels(
            levels,
            weights,
            aggregateby,
            misc_cell_props_col=self._misc_cell_props_col,
            **kwargs,
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
            indices, misc_cell_props_col=self._misc_cell_props_col, **kwargs
        )
        if not self.cell_properties.empty:
            cell_properties = self.cell_properties.loc[
                list(restricted.uidset)
            ].reset_index()
            restricted.assign_cell_properties(cell_properties)
        return restricted

    def assign_cell_properties(
        self,
        cell_props: pd.DataFrame | dict[T, dict[T, dict[Any, Any]]],
        misc_col: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Assign new properties to cells of the incidence matrix and update
        :attr:`properties`

        Parameters
        ----------
        cell_props : pandas.DataFrame, dict of iterables, or doubly-nested dict, optional
            See documentation of the `cell_properties` parameter in :class:`EntitySet`
        misc_col: str, optional
            name of column to be used for miscellaneous cell property dicts
        replace: bool, default=False
            If True, replace existing :attr:`cell_properties` with result;
            otherwise update with new values from result

        Raises
        -----
        AttributeError
            Not supported for :attr:`dimsize`=1
        """
        if self.dimsize < 2:
            raise AttributeError(
                f"cell properties are not supported for 'dimsize'={self.dimsize}"
            )

        misc_col = misc_col or self._misc_cell_props_col
        try:
            cell_props = cell_props.rename(
                columns={misc_col: self._misc_cell_props_col}
            )
        except AttributeError:  # handle cell props in nested dict format
            self._cell_properties_from_dict(cell_props)
        else:  # handle cell props in DataFrame format
            self._cell_properties_from_dataframe(cell_props)

    def _cell_properties_from_dataframe(self, cell_props: pd.DataFrame) -> None:
        """Private handler for updating :attr:`properties` from a DataFrame

        Parameters
        ----------
        props

        Parameters
        ----------
        cell_props : DataFrame
        """
        if cell_props.index.nlevels > 1:
            extra_levels = [
                idx_lev
                for idx_lev in cell_props.index.names
                if idx_lev not in self._data_cols
            ]
            cell_props = cell_props.reset_index(level=extra_levels)

        misc_col = self._misc_cell_props_col

        try:
            cell_props.index = cell_props.index.reorder_levels(self._data_cols)
        except AttributeError:
            if cell_props.index.name in self._data_cols:
                cell_props = cell_props.reset_index()

            try:
                cell_props = cell_props.set_index(
                    self._data_cols, verify_integrity=True
                )
            except ValueError:
                warnings.warn(
                    "duplicate cell rows will be dropped after first occurrence"
                )
                cell_props = cell_props.drop_duplicates(self._data_cols)
                cell_props = cell_props.set_index(self._data_cols)

        if misc_col in cell_props:
            try:
                cell_props[misc_col] = cell_props[misc_col].apply(literal_eval)
            except ValueError:
                pass  # data already parsed, no literal eval needed
            else:
                warnings.warn("parsed cell property dict column from string literal")

        cell_properties = cell_props.combine_first(self.cell_properties)
        # import ipdb; ipdb.set_trace()
        # cell_properties[misc_col] = self.cell_properties[misc_col].combine(
        #     cell_properties[misc_col],
        #     lambda x, y: {**(x if pd.notna(x) else {}), **(y if pd.notna(y) else {})},
        #     fill_value={},
        # )

        self._cell_properties = cell_properties.sort_index()

    def _cell_properties_from_dict(
        self, cell_props: dict[T, dict[T, dict[Any, Any]]]
    ) -> None:
        """Private handler for updating :attr:`cell_properties` from a doubly-nested dict

        Parameters
        ----------
        cell_props
        """
        # TODO: there may be a more efficient way to convert this to a dataframe instead
        #  of updating one-by-one via nested loop, but checking whether each prop_name
        #  belongs in a designated existing column or the misc. property dict column
        #  makes it more challenging.
        #  For now: only use nested loop update if non-misc. columns currently exist
        if len(self.cell_properties.columns) > 1:
            for item1 in cell_props:
                for item2 in cell_props[item1]:
                    for prop_name, prop_val in cell_props[item1][item2].items():
                        self.set_cell_property(item1, item2, prop_name, prop_val)
        else:
            cells = pd.MultiIndex.from_tuples(
                [(item1, item2) for item1 in cell_props for item2 in cell_props[item1]],
                names=self._data_cols,
            )
            props_data = [cell_props[item1][item2] for item1, item2 in cells]
            cell_props = pd.DataFrame(
                {self._misc_cell_props_col: props_data}, index=cells
            )
            self._cell_properties_from_dataframe(cell_props)

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
            .groupby(self._data_cols[0], as_index=False, observed=False)
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

    def set_cell_property(
        self, item1: T, item2: T, prop_name: Any, prop_val: Any
    ) -> None:
        """Set a property of a cell i.e., incidence between items of different levels

        Parameters
        ----------
        item1 : hashable
            name of an item in level 0
        item2 : hashable
            name of an item in level 1
        prop_name : hashable
            name of the cell property to set
        prop_val : any
            value of the cell property to set

        See Also
        --------
        get_cell_property, get_cell_properties
        """
        if item2 in self.elements[item1]:
            if prop_name in self.properties:
                self._cell_properties.loc[(item1, item2), prop_name] = pd.Series(
                    [prop_val]
                )
            else:
                try:
                    self._cell_properties.loc[
                        (item1, item2), self._misc_cell_props_col
                    ].update({prop_name: prop_val})
                except KeyError:
                    self._cell_properties.loc[(item1, item2), :] = {
                        self._misc_cell_props_col: {prop_name: prop_val}
                    }

    def get_cell_property(self, item1: T, item2: T, prop_name: Any) -> Any:
        """Get a property of a cell i.e., incidence between items of different levels

        Parameters
        ----------
        item1 : hashable
            name of an item in level 0
        item2 : hashable
            name of an item in level 1
        prop_name : hashable
            name of the cell property to get

        Returns
        -------
        prop_val : any
            value of the cell property

        See Also
        --------
        get_cell_properties, set_cell_property
        """
        try:
            cell_props = self.cell_properties.loc[(item1, item2)]
        except KeyError:
            raise
            # TODO: raise informative exception

        try:
            prop_val = cell_props.loc[prop_name]
        except KeyError:
            prop_val = cell_props.loc[self._misc_cell_props_col].get(prop_name)

        return prop_val

    def get_cell_properties(self, item1: T, item2: T) -> dict[Any, Any]:
        """Get all properties of a cell, i.e., incidence between items of different
        levels

        Parameters
        ----------
        item1 : hashable
            name of an item in level 0
        item2 : hashable
            name of an item in level 1

        Returns
        -------
        dict
            ``{named cell property: cell property value, ..., misc. cell property column
            name: {cell property name: cell property value}}``

        See Also
        --------
        get_cell_property, set_cell_property
        """
        try:
            cell_props = self.cell_properties.loc[(item1, item2)]
        except KeyError:
            raise
            # TODO: raise informative exception

        return cell_props.to_dict()
