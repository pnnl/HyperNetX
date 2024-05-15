import pandas as pd

from hypernetx import HyperNetXError


def mkdict(x):
    # function to create a dictionary from object x if it is not already a dictionary.
    import ast, json

    if isinstance(x, dict):
        return x
    else:
        try:
            temp = ast.literal_eval(x)
        except:
            try:
                temp = json.loads(x)
            except:
                temp = {}
        if isinstance(temp, dict):
            return temp
        else:
            return {}


def create_df(
    dfp,
    uid_cols=None,
    level=0,
    use_index=False,
    weight_prop=None,
    default_weight=1.0,
    misc_properties_col=None,
    aggregation_methods=None,
):

    if not isinstance(dfp, pd.DataFrame):
        raise TypeError("method requires a Pandas DataFrame")
    else:
        # checks if the use index variable is called. if it is then use the existing indices. if it is not then an index is set based on the uid columns.
        if use_index == False:
            # if uid cols are specified make those columns the index columns
            if uid_cols != None:
                # create chk function to check if the column specified is a string. if it is not a string then it assumes it is an integer and grabs that columns name.
                chk = lambda c: c if isinstance(c, str) else dfp.columns[c]
                # set indices using the column names in uid_cols using the chk function.
                dfp = dfp.set_index([chk(c) for c in uid_cols])
            else:  # if uid_cols are not specified then assume the first one or two columns (depending on level) are the index columns and set the index.
                if level == 2:
                    dfp = dfp.set_index([dfp.columns[0], dfp.columns[1]])
                else:
                    dfp = dfp.set_index([dfp.columns[0]])

        # if the misc prop col is in the column names
        if misc_properties_col in dfp.columns:
            # rename the misc properties column to the default name if it isn't
            if misc_properties_col != "misc_properties":
                dfp = dfp.rename(columns={misc_properties_col: "misc_properties"})
            # force misc properties to be a dictionary if it is not.
            dfp.misc_properties = dfp.misc_properties.map(mkdict)
        else:  # if the column is not specified then create the misc properties column of empty dicitonaries.
            dfp["misc_properties"] = [{} for row in dfp.index]

        # check if weight property column name was specified.
        if weight_prop in dfp.columns:
            # if it was specified and it exists then rename to default weight name and fill in the NA weights with the default.
            dfp = dfp.rename(columns={weight_prop: "weight"})
            dfp = dfp.fillna({"weight": default_weight})
        # if weight column is not None and the weight column name was not in the column names then check in the misc properties.
        elif weight_prop is not None:

            def grabweight(cell):
                # function to grab weights from the misc properties column.
                if isinstance(cell, dict):
                    return cell.get(weight_prop, default_weight)
                else:
                    return default_weight

            # set the weight column to the weights grabbed from the misc properties dictionary (if any).
            dfp["weight"] = dfp["misc_properties"].map(grabweight)

    # reorder columns in standard order
    cols = [c for c in dfp.columns if c not in ["weight", "misc_properties"]]
    dfp = dfp[["weight"] + cols + ["misc_properties"]]

    # remove duplicate indices and aggregate using aggregation methods specified.
    dfp = dfp[~dfp.index.duplicated(keep="first")]

    # rename index columns if necessary
    if level == 0 or level == 1:
        # rename index column to 'uid'
        dfp.index.names = ["uid"]
    elif level == 2:
        # rename index columns to 'edges' and 'nodes'
        dfp.index.names = ["edges", "nodes"]

    return dfp


def dataframe_factory_method(
    DF,
    level,
    use_indices=False,
    uid_cols=None,
    misc_properties_col="misc_properties",
    weight_col="weight",
    default_weight=1.0,
    aggregate_by={},
):
    """
    This function creates a pandas dataframe in the correct format given a
    pandas dataframe of either cell, node, or edge properties.

    Parameters
    ----------

    DF : dataframe
        dataframe of properties for either incidences, edges, or nodes

    level : int
        Level to specify the type of data the dataframe is for: 0 for edges, 1 for nodes, and 2 for incidences (cells).

    uid_cols : list of str or int
        column index (or name) in pandas.dataframe
        used for (hyper)edge, node, or incidence (edge, node) IDs.


    misc_properties_col : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    weight_col : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    default_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    aggregate_by : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum
        the weights of the aggregated duplicate rows.

    Returns
    -------
    Pandas Dataframe of the property store in the correct format for HNX.

    """

    if DF is None:  # if no properties are provided for that property type.
        PS = None

    else:
        PS = create_df(
            DF,
            uid_cols=uid_cols,
            level=level,
            use_index=use_indices,
            weight_prop=weight_col,
            misc_properties_col=misc_properties_col,
            default_weight=default_weight,
            aggregation_methods=aggregate_by,
        )

    return PS


def dict_to_incidence_store_df(D):
    L0 = []  # list of keys
    L1 = []  # list of the values
    for edge in D:
        nodes = D[edge]
        for node in nodes:
            L0.append(edge)
            L1.append(node)
    return pd.DataFrame(
        {
            "level_0": L0,
            "level_1": L1,
        }
    )


def dict_factory_method(
    D,
    level,
    use_indices=False,
    uid_cols=None,
    misc_properties_col="misc_properties",
    weight_col="weight",
    default_weight=1.0,
    aggregate_by={},
):
    '''
    This function creates a pandas dataframe in the correct format given a
    dictionary of either cell, node, or edge properties.

    Parameters
    ----------

    D : dictionary
        dictionary of properties for either incidences, edges, or nodes

    level : int
        Level to specify the type of data the dataframe is for: 0 for edges, 1 for nodes, and 2 for incidences (cells).

    uid_cols : list of str or int
        column index (or name) in pandas.dataframe
        used for (hyper)edge, node, or incidence (edge, node) IDs.

    misc_properties_col : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    weight_col : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    default_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    aggregate_by : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum
        the weights of the aggregated duplicate rows.

    """

    Returns
    -------
    Pandas Dataframe of the property store in the correct format for HNX.

    '''

    # if no dictionary is provided set it to an empty dictionary.
    if D is None:
        DF = None
    # if the dictionary data provided is for the setsystem (incidence data)
    elif level == 2:
        # explode list of lists into incidence pairs as a pandas dataframe using pandas series explode.
        # DF = pd.DataFrame(pd.Series(D).explode()).reset_index()
        DF = dict_to_incidence_store_df(D)
        # rename columns to correct column names for edges and nodes
        DF = DF.rename(columns=dict(zip(DF.columns, ["edges", "nodes"])))
        attribute_data = {weight_col: [], misc_properties_col: []}
        for _, incidence_pair in DF.iterrows():
            edge, node = incidence_pair
            if isinstance(D[edge], dict):
                attributes_of_incidence_pair = D[edge][node]
                if weight_col in attributes_of_incidence_pair:
                    weight_val = attributes_of_incidence_pair.pop(weight_col)
                    attribute_data[weight_col] += [weight_val]
                else:
                    attribute_data[weight_col] += [default_weight]
                attribute_data[misc_properties_col] += [attributes_of_incidence_pair]
        attribute_df = pd.DataFrame(attribute_data)
        DF = pd.concat([DF, attribute_df], axis=1)

    # if the dictionary is for edges or nodes.
    elif level == 1 or level == 0:
        attribute_data = {weight_col: [], misc_properties_col: []}
        for data_uid in D.values():
            if isinstance(data_uid, dict):
                attributes_of_uid = data_uid
                if weight_col in attributes_of_uid:
                    weight_val = attributes_of_uid.pop(weight_col)
                    attribute_data[weight_col] += [weight_val]
                else:
                    attribute_data[weight_col] += [default_weight]
                attribute_data[misc_properties_col] += [attributes_of_uid]

        attribute_df = pd.DataFrame(attribute_data)
        DF = pd.concat([pd.DataFrame(list(D.keys())), attribute_df], axis=1)

    # get property store from dataframe
    PS = dataframe_factory_method(
        DF,
        level=level,
        use_indices=use_indices,
        uid_cols=uid_cols,
        misc_properties_col=misc_properties_col,
        weight_col=weight_col,
        default_weight=default_weight,
        aggregate_by=aggregate_by,
    )

    return PS


def list_factory_method(
    L,
    level,
    use_indices=False,
    uid_cols=None,
    misc_properties_col="misc_properties",
    weight_col="weight",
    default_weight=1.0,
    aggregate_by={},
):
    '''

    This function creates a pandas dataframe in the correct format given a
    list of lists to be used as the cell property store dataframe.

    Parameters
    ----------

    L : list of lists
        list of lists representing the nodes in each hyperedge.

    level : int
        Level to specify the type of data the dataframe is for: 0 for edges, 1 for nodes, and 2 for incidences (cells).

    uid_cols : list of str or int
        column index (or name) in pandas.dataframe
        used for (hyper)edge, node, or incidence (edge, node) IDs.

    misc_properties_col : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    weight_col : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    default_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    aggregate_by : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum
        the weights of the aggregated duplicate rows.

    """

    Returns
    -------
    Pandas Dataframe of the property store in the correct format for HNX.
    '''

    if L is None:
        PS = None
    else:
        # explode list of lists into incidence pairs as a pandas dataframe using pandas series explode.
        DF = pd.DataFrame(pd.Series(L).explode()).reset_index()
        # rename columns to correct column names for edges and nodes
        DF = DF.rename(columns=dict(zip(DF.columns, ["edges", "nodes"])))
        # create property store from dataframe.
        PS = dataframe_factory_method(
            DF,
            level=level,
            use_indices=use_indices,
            uid_cols=uid_cols,
            misc_properties_col=misc_properties_col,
            weight_col=weight_col,
            default_weight=default_weight,
            aggregate_by=aggregate_by,
        )

    return PS


def ndarray_factory_method(arr, level, *args, **kwargs):
    shape = arr.shape
    if len(shape) == 2 and shape[1] == 2 and level == 2:
        return dataframe_factory_method(pd.DataFrame(arr), 2, *args, **kwargs)
    raise HyperNetXError("An ndarray of shape (N,2) can only be used as a setsystem")
