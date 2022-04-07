import pandas as pd

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
        )

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


def assign_weights(df, weights=None, weight_col="cell_weights"):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to assign a weight column to
    weights : array-like or Hashable, optional
        If numpy.ndarray with the same length as df, create a new weight column with
        these values.
        If Hashable, must be the name of a column of df to assign as the weight column
        Otherwise, create a new weight column assigning a weight of 1 to every row
    weight_col : Hashable
        Name for new column if one is created (not used if the name of an existing
        column is passed as weights)

    Returns
    -------
    df : pandas.DataFrame
        The original DataFrame with a new column added if needed
    weight_col : str
        Name of the column assigned to hold weights
    """
    if isinstance(weights, (list, np.ndarray)) and len(weights) == len(df):
        df[weight_col] = weights
    elif isinstance(weights, Hashable) and weights in df:
        weight_col = weights
    else:
        df[weight_col] = np.ones(len(df), dtype=int)

    return df, weight_col


def remove_row_duplicates(df, data_cols, weights=None, aggregateby="sum"):
    """
    Removes and aggregates duplicate rows of a DataFrame using groupby

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame to remove or aggregate duplicate rows from
    data_cols : list
        A list of column names in df to perform the groupby on / remove duplicates from
    weights : array-like or Hashable, optional
        Argument passed to assign_weights
    aggregateby : str, optional, default='sum'
        A valid aggregation method for pandas groupby
        If None, drop duplicates without aggregating weights

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with duplicate rows removed or aggregated
    weight_col : Hashable
        The name of the column holding aggregated weights, or None if aggregateby=None
    """
    df = df.copy()
    categories = {}
    for col in data_cols:
        if df[col].dtype.name == "category":
            categories[col] = df[col].cat.categories
            df[col] = df[col].astype(categories[col].dtype)

    if not aggregateby:
        df = df.drop_duplicates(subset=data_cols)

    df, weight_col = assign_weights(df, weights=weights)

    if aggregateby:
        df = df.groupby(data_cols, as_index=False, sort=False).agg(
            {weight_col: aggregateby}
        )

    for col in categories:
        df[col] = df[col].astype(CategoricalDtype(categories=categories[col]))

    return df, weight_col
