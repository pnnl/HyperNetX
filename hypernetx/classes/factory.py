import pandas as pd
import numpy as np
from hypernetx.classes.helpers import dict_depth


# In[ ]:
# -------------------------------------------------------------------------------------------------
# Individual factory methods for property stores
# -------------------------------------------------------------------------------------------------


def remove_property_store_duplicates(PS, default_uid_col_names, aggregation_methods={}):
    agg_methods = {}
    for col in PS.columns:
        if col not in aggregation_methods:
            agg_methods[col] = "first"
        else:
            agg_methods[col] = aggregation_methods[col]
    return PS.groupby(level=default_uid_col_names).agg(agg_methods)


### Alternate code for creating dataframe for PS
import ast, json


def mkdict(x):
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
        # dfp = deepcopy(properties)   ### not sure if this is wise

        if use_index == False:
            if uid_cols != None:
                chk = lambda c: c if isinstance(c, str) else dfp.columns[c]
                dfp = dfp.set_index([chk(c) for c in uid_cols])
            else:
                if level == 2:
                    dfp = dfp.set_index(dfp.columns[0], dfp.columns[1])
                else:
                    dfp = dfp.set_index(dfp.columns[0])

        if (
            misc_properties_col in dfp.columns
            and misc_properties_col != "misc_properties"
        ):
            dfp = dfp.rename(columns={misc_properties_col: "misc_properties"})
            dfp.misc_properties = dfp.misc_properties.map(mkdict)
        else:
            dfp["misc_properties"] = [{} for row in dfp.index]

        if weight_prop in dfp.columns:
            dfp = dfp.rename(columns={weight_prop: "weight"})
            dfp = dfp.fillna({"weight": default_weight})
        elif weight_prop is not None:

            def grabweight(cell):
                if isinstance(cell, dict):
                    return cell.get(weight_prop, default_weight)
                else:
                    return default_weight

            dfp["weight"] = dfp["misc_properties"].map(grabweight)

    cols = [c for c in dfp.columns if c not in ["weight", "misc_properties"]]
    dfp = dfp[["weight"] + cols + ["misc_properties"]]
    dfp = dfp[~dfp.index.duplicated(keep="first")]
    return dfp


# def create_df(properties, uid_cols, use_indices,
#               default_uid_col_names, weight_prop_col,
#               misc_prop_col, default_weight, aggregation_methods):

#     #get length of dataframe once to be used throughout this function.
#     length_of_dataframe = len(properties)

#     #get column names if integer was provided instead
#     if isinstance(weight_prop_col, int):
#         weight_prop_col = properties.columns[weight_prop_col]
#     if isinstance(misc_prop_col, int):
#         misc_prop_col = properties.columns[misc_prop_col]

#     #get list of all column names in properties dataframe
#     column_names = list(properties.columns)


#     # set weight column code:
#     # default to use weight column if it exists before looking for default weight array or in misc properties column.
#     if weight_prop_col in column_names:
#         #do nothing since this is the format we expect by default.
#         pass
#     #check to see if an array of weights was provided to use for weights column
#     elif not isinstance(default_weight, int) and not isinstance(default_weight, float):
#         properties[weight_prop_col] = default_weight

#     #check if the weight column name exists in the misc properties.
#     elif misc_prop_col in column_names: #check if misc properties exists
#         #check if weight_prop_col is a key in any of the misc properties dicitonary.
#         if any(weight_prop_col in misc_dict for misc_dict in properties[misc_prop_col]):
#             #create list of cell weights from misc properties dictionaries and use default value if not in keys
#             weights_from_misc_dicts = []
#             for misc_dict in properties[misc_prop_col]:
#                 if weight_prop_col in misc_dict:
#                     weights_from_misc_dicts.append(misc_dict[weight_prop_col])
#                 else:
#                     weights_from_misc_dicts.append(default_weight)
#             properties[weight_prop_col] = weights_from_misc_dicts

#     #if not provided anywhere then add in as default value
#     else:
#         properties[weight_prop_col] = [default_weight]*length_of_dataframe

#     #rename the columns where needed
#     #start by defining dictionary of column renaming with uid columns.
#     if not use_indices: #include uid columns if they are not indices.
#         col_rename_dict = {uid_cols[i]: default_uid_col_names[i] for i in range(len(uid_cols))} #renaming dictionary
#     else:
#         col_rename_dict = {}
#     #add weight column renaming
#     col_rename_dict[weight_prop_col] = 'weight'
#     #set misc properties column if not already provided and if set then update renaming dictionary.
#     if misc_prop_col not in column_names:
#         properties['misc_properties'] = [{}]*length_of_dataframe
#     else:
#         col_rename_dict[misc_prop_col] = 'misc_properties'
#     #rename the columns
#     properties.rename(columns = col_rename_dict, inplace = True) #rename the columns


#     #set index for dataframe using the default uid column names that are dependent on the level if indices flag not on.
#     if not use_indices:
#         properties = properties.set_index(default_uid_col_names)
#     else: #otherwise just rename the incides to the default names.
#         properties.index.names = default_uid_col_names


#     #remove any NaN values or missing values in weight column
#     properties['weight'].fillna(default_weight, inplace = True)


#     # remove any duplicate indices and combine using aggregation methods (defaults to 'first' if none provided).
#     properties = remove_property_store_duplicates(properties, default_uid_col_names, aggregation_methods = aggregation_methods)


#     #reorder columns to have properties last
#     # Get the column names and the specific column
#     specific_col = 'misc_properties'
#     # Create a new order for the columns
#     updated_column_names = list(properties.columns)
#     new_order = [col for col in updated_column_names if col != specific_col] + [specific_col]
#     # Reorder the dataframe using reindex
#     properties = properties.reindex(columns=new_order)

#     return properties


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
        if use_indices:
            uid_cols = DF.index.names
        else:
            # uid column name setting if they are not provided
            if (
                uid_cols is None
            ):  # if none are provided set to the names of the first or first two columns depending on level
                if level == 0 or level == 1:
                    uid_cols = [DF.columns[0]]
                elif level == 2:
                    uid_cols = [DF.columns[0], DF.columns[1]]

            # get column names if integer was provided instead and create new uid_cols with string names.
            uid_cols_to_str = []
            for col in uid_cols:
                if isinstance(col, int):
                    uid_cols_to_str.append(DF.columns[col])
                else:
                    uid_cols_to_str.append(col)
            uid_cols = uid_cols_to_str

        # set default uid column name(s)
        if level == 0 or level == 1:
            default_uid_col_names = ["uid"]
        elif level == 2:
            default_uid_col_names = ["edges", "nodes"]

        # PS = create_df(DF, uid_cols = uid_cols, use_indices = use_indices,
        #                default_uid_col_names = default_uid_col_names,
        #                weight_prop_col = weight_col,
        #                misc_prop_col = misc_properties_col,
        #                default_weight = default_weight,
        #                aggregation_methods = aggregate_by)

        PS = create_df(
            DF,
            uid_cols=uid_cols,
            use_index=use_indices,
            weight_prop=weight_col,
            misc_properties_col=misc_properties_col,
            default_weight=default_weight,
            aggregation_methods=aggregate_by,
        )

    return PS


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
        DF = pd.DataFrame(pd.Series(D).explode()).reset_index()
        # rename columns to correct column names for edges and nodes
        DF = DF.rename(columns=dict(zip(DF.columns, ["edges", "nodes"])))

        # if attributes are stored on the dictionary (ie, it has a depth greater than 2)
        if dict_depth(D) > 2:
            attribute_data = []
            for _, incidence_pair in DF.iterrows():
                edge, node = incidence_pair
                attributes_of_incidence_pair = D[edge][node]
                attribute_data.append(attributes_of_incidence_pair)
            attribute_df = pd.DataFrame(attribute_data)
            DF = pd.concat([DF, attribute_df], axis=1)

    else:
        attribute_data = []
        for key in D:
            attributes_of_key = D[key]
            attribute_data.append(attributes_of_key)
        attribute_df = pd.DataFrame(attribute_data)
        DF = pd.concat(
            [pd.DataFrame(list(D.keys()), columns=uid_cols), attribute_df], axis=1
        )

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


"""
# In[ ]: testing code
# Only runs if running from this file (This will show basic examples and testing of the code)


if __name__ == "__main__":

    run_list_example = False
    if run_list_example:

        list_of_iterables = [[1, 1, 2], {1, 2}, {1, 2, 3}]
        display(list_of_iterables)

        IPS = list_factory_method(list_of_iterables, level = 2,
                                  aggregate_by = {'weight': 'sum'})
        display(IPS)
        print('-'*100)



    run_simple_dict_example = True
    if run_simple_dict_example:

        cell_dict = {'e1':[1,2],'e2':[1,2],'e3':[1,2,3]}

        print('Provided Dataframes')
        print('-'*100)
        display(cell_dict)

        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)

        IPS = dict_factory_method(cell_dict, level = 2)

        display(IPS)
        print('-'*100)


    run_dict_example = True
    if run_dict_example:

        cell_prop_dict = {'e1':{ 1: {'w':0.5, 'name': 'related_to'},
                                 2: {'w':0.1, 'name': 'related_to','startdate': '05.13.2020'}},
                          'e2':{ 1: {'w':0.52, 'name': 'owned_by'},
                                 2: {'w':0.2}},
                          'e3':{ 1: {'w':0.5, 'name': 'related_to'},
                                 2: {'w':0.2, 'name': 'owner_of'},
                                 3: {'w':1, 'type': 'relationship'}}}

        edge_prop_dict = {'e1': {'number': 1},
                          'e2': {'number': 2},
                          'e3': {'number': 3}}

        print('Provided Dataframes')
        print('-'*100)
        display(cell_prop_dict)

        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)

        IPS = dict_factory_method(cell_prop_dict, level = 2, weight_col = 'w')
        display(IPS)


        EPS = dict_factory_method(edge_prop_dict, level = 0)
        display(EPS)


        NPS = dict_factory_method(None, level = 1, weight_col = 'w')
        display(NPS)
        print('-'*100)


    run_simple_dataframe_example = False
    if run_simple_dataframe_example:

        incidence_dataframe = pd.DataFrame({'e': ['a', 'a', 'a', 'b', 'c', 'c'], 'n': [1, 1, 2, 3, 2, 3],})


        print('Provided Dataframes')
        print('-'*100)
        display(incidence_dataframe)



        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)



        IPS = dataframe_factory_method(incidence_dataframe, level = 2,
                                       uid_cols = ['e', 'n'],
                                       aggregate_by = {'weight': 'sum'},)
        IS = IPS.index

        display(IS)
        display(IPS)

        EPS = dataframe_factory_method(None, level = 0)
        display(EPS)

        NPS = dataframe_factory_method(None, level = 1, uid_cols = ['nodes'])
        display(NPS)
        print('-'*100)


    run_dataframe_example = True
    if run_dataframe_example:
        print('')
        print('='*100)
        print('='*100)
        print('='*100)
        print('')

        cell_prop_dataframe = pd.DataFrame({'E': ['a', 'a', 'a', 'b', 'c', 'c'], 'nodes': [1, 1, 2, 3, 2, 3],
                                            'color': ['red', 'red', 'red', 'red', 'red', 'blue'],
                                            'other_properties': [{}, {}, {'weight': 5}, {'time': 3}, {}, {}]})

        edge_prop_dataframe = pd.DataFrame({'edges': ['a', 'b', 'c'],
                                            'strength': [2, np.nan, 3]})

        node_prop_dataframe = pd.DataFrame({'N': [1],
                                            'temperature': [60]})
        node_prop_dataframe.set_index(['N'], inplace = True)

        print(list(node_prop_dataframe.columns))

        print('Provided Dataframes')
        print('-'*100)
        display(cell_prop_dataframe)
        display(edge_prop_dataframe)
        display(node_prop_dataframe)

        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)


        IPS = dataframe_factory_method(cell_prop_dataframe, level = 2,
                                       uid_cols = ['E', 'nodes'],
                                       misc_properties_col = 'other_properties',
                                       aggregate_by = {'weight': 'sum'},)
        IS = IPS.index

        display(IS)

        display(IPS)


        EPS = dataframe_factory_method(edge_prop_dataframe, level = 0,
                                       weight_col = 1, uid_cols = [0])
        display(EPS)


        NPS = dataframe_factory_method(node_prop_dataframe, level = 1,
                                       use_indices = True)
        display(NPS)
        print('-'*100)
"""
