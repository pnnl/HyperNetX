

import pandas as pd
import networkx as nx
import numpy as np


# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for property stores
#-------------------------------------------------------------------------------------------------


def remove_property_store_duplicates(PS, default_uid_cols, aggregation_methods = {}):
    for col in PS.columns:
        if col not in aggregation_methods:
            aggregation_methods[col] = 'first'
    return PS.groupby(level = default_uid_cols).agg(aggregation_methods)


def create_df(properties, uid_cols, indices, multi_index,
              default_uid_cols, weight_prop_col,
              misc_prop_col, default_weight, aggregation_methods):

    # initialize a dictionary to be converted to a pandas dataframe.
    properties_df_dict = {}

    #get names of property columns provided that are not the edge or node columns
    property_columns = set(list(properties.columns)) - set(uid_cols)
    for prop in property_columns: #set those as rows in DF
        properties_df_dict[prop] = properties.loc[:, prop]


    #get column names if integer was provided instead and create new uid_cols with string names.
    uid_cols_to_str = []
    for col in uid_cols:
        if isinstance(col, int):
            uid_cols_to_str.append(properties.columns[col])
        else:
            uid_cols_to_str.append(col)
    uid_cols = uid_cols_to_str
    if isinstance(weight_prop_col, int):
        weight_prop_col = properties.columns[weight_prop_col]
    if isinstance(misc_prop_col, int):
        misc_prop_col = properties.columns[misc_prop_col]


    #rename uid columns if needed to default names. No way of doing this correctly without knowing data type.
    for i in range(len(uid_cols)):
        col, default_col = uid_cols[i], default_uid_cols[i]
        #change name of edges column to "edges"
        if col != default_col and col in properties_df_dict:
            # Pop the value associated with col key and store it
            column_values = properties_df_dict.pop(col)
            # Add the popped value with the correct col name
            properties_df_dict[default_col] = column_values


    # set weight column code:
    # check if weight column exists or if weight col name exists in dictionary and assign it as column if it doesn't
    # default to use weight column if it exists before looking in misc properties column.
    if weight_prop_col not in property_columns:
        create_default_weight_column = False #start by default not setting weight column assuming it exists.
        if misc_prop_col in property_columns: #check if misc properties exists
            #make sure that an array of weights wasn't provided in default_weight. use that if it was.
            if not (isinstance(default_weight, int) or isinstance(default_weight, float)):
                #check if weight_prop_col is a key in any of the misc properties dicitonary.
                if any(weight_prop_col in misc_dict for misc_dict in properties_df_dict[misc_prop_col]):
                    #create list of cell weights from misc properties dictionaries and use default value if not in keys
                    weights_from_misc_dicts = []
                    for misc_dict in properties_df_dict[misc_prop_col]:
                        if weight_prop_col in misc_dict:
                            weights_from_misc_dicts.append(misc_dict[misc_prop_col])
                        else:
                            weights_from_misc_dicts.append(default_weight)
                    properties_df_dict[weight_prop_col] = weights_from_misc_dicts
                else:
                    create_default_weight_column = True
            else:
                create_default_weight_column = True
        else:
            create_default_weight_column = True

        if create_default_weight_column:
            if isinstance(default_weight, int) or isinstance(default_weight, float):
                properties_df_dict[weight_prop_col] = [default_weight]*len(indices)
            else:
                properties_df_dict[weight_prop_col] = default_weight

    #change name of weight props column to "weight"
    if weight_prop_col != 'weight':
        # Pop the value associated with wieght key and store it
        column_values = properties_df_dict.pop(weight_prop_col)
        # Add the popped value with the new weight name "weight"
        properties_df_dict['weight'] = column_values


    #set misc properties column if not already provided
    if misc_prop_col not in property_columns:
        properties_df_dict[misc_prop_col] = [{}]*len(indices)
    #change name of misc props column to "properties"
    if misc_prop_col != 'misc_properties':
        # Pop the value associated with properties key and store it
        misc_props = properties_df_dict.pop(misc_prop_col)
        # Add the popped value with the new properties name "misc_properties"
        properties_df_dict['misc_properties'] = misc_props


    #create dataframe from property store dictionary object
    PS = pd.DataFrame(properties_df_dict)
    #set multi index for dataframe
    PS = PS.set_index(multi_index)

    #remove any NaN values or missing values in weight column
    PS['weight'].fillna(default_weight, inplace = True)


    # remove any duplicate indices and combine using aggregation methods (defaults to 'first' if none provided).
    PS = remove_property_store_duplicates(PS, default_uid_cols, aggregation_methods = aggregation_methods)

    #reorder columns to have properties last
    # Get the column names and the specific column
    column_names = list(PS.columns)
    specific_col = 'misc_properties'
    # Create a new order for the columns
    new_order = [col for col in column_names if col != specific_col] + [specific_col]
    # Reorder the dataframe using reindex
    PS = PS.reindex(columns=new_order)

    return PS



def property_store_from_dataframe(properties, property_type,
                                     edge_col = 'edges', node_col = 'nodes',
                                     misc_cell_properties = 'misc_properties', misc_properties = 'misc_properties',
                                     weight_prop = None, edge_weight_prop = 'weight', node_weight_prop = 'weight',
                                     cell_weight_col = 'weight',
                                     cell_weights = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0,
                                     aggregation_methods = {}):



    """
    This function creates a pandas dataframe in the correct format given a
    pandas dataframe of either cell, node, or edge properties.

    Parameters
    ----------

    properties : dataframe
        dataframe of properties for either incidences, edges, or nodes

    property_type : str
        type of property dataframe provided.
        Either 'edge_properties', 'node_properties', or 'cell_properties'

    edge_col : (optional) str | int, default = 0
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for (hyper)edge ids. Will be used to reference edgeids for
        all set systems.

    node_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for node ids. Will be used to reference nodeids for all set systems.

    cell_weight_col : (optional) str | int, default = None
        column index (or name) in pandas.dataframe or numpy.ndarray used for
        referencing cell weights. For a dict of dicts references key in cell
        property dicts.

    cell_weights : (optional) Sequence[float,int] | int |  float , default = 1.0
        User specified cell_weights or default cell weight.
        Sequential values are only used if setsystem is a
        dataframe or ndarray in which case the sequence must
        have the same length and order as these objects.
        Sequential values are ignored for dataframes if cell_weight_col is already
        a column in the data frame.
        If cell_weights is assigned a single value
        then it will be used as default for missing values or when no cell_weight_col
        is given.

    misc_cell_properties : (optional) str | int, default = None
        Column name of dataframe corresponding to a column of variable
        length property dictionaries for the cell. Ignored for other setsystem
        types.

    misc_properties : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    edge_weight_prop : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    node_weight_prop : (optional) str, default = None,
        Name of property in node_properties to use for weight.

    weight_prop : (optional) str, default = None
        Name of property in properties to use for 'weight'

    default_edge_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.

    default_node_weight : (optional) int | float, default = 1
        Used when node weight property is missing or undefined


    aggregation_methods : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum
        the weights of the aggregated duplicate rows.

    """

    if properties is None: #if no properties are provided for that property type.
        if property_type == 'cell_properties':
            default_uid_cols = ['edges', 'nodes']
        elif property_type == 'edge_properties':
            default_uid_cols = ['edges']
        elif property_type == 'node_properties':
            default_uid_cols = ['nodes']
        multi_index = pd.MultiIndex.from_tuples(levels=[[],[]], codes=[[],[]], names=default_uid_cols)
        PS = pd.DataFrame(index = multi_index, columns=['weight', 'misc_properties'])

    else:
        if property_type == 'cell_properties':
            incidence_pairs = np.array(properties[[edge_col, node_col]]) #array of incidence pairs to use as UIDs.
            indices = [tuple(incidence_pair) for incidence_pair in incidence_pairs]
            multi_index = pd.MultiIndex.from_tuples(indices, names=['edges', 'nodes'])
            default_uid_cols = ['edges', 'nodes']
            PS = create_df(properties,
                           uid_cols = [edge_col, node_col],
                           default_uid_cols = default_uid_cols,
                           indices = indices, multi_index = multi_index,
                           weight_prop_col = cell_weight_col,
                           misc_prop_col = misc_cell_properties,
                           default_weight = cell_weights,
                           aggregation_methods = aggregation_methods)

        elif property_type == 'edge_properties':
            edge_names = np.array(properties[[edge_col]]) #array of edges to use as UIDs.
            indices = [tuple([edge_name[0]]) for edge_name in edge_names]
            multi_index = pd.MultiIndex.from_tuples(indices, names=['edges'])
            default_uid_cols = ['edges']
            PS = create_df(properties,
                           uid_cols = [edge_col],
                           default_uid_cols = default_uid_cols,
                           indices = indices, multi_index = multi_index,
                           weight_prop_col = edge_weight_prop,
                           misc_prop_col = misc_properties,
                           default_weight = default_edge_weight,
                           aggregation_methods = aggregation_methods)

        elif property_type == 'node_properties':
            node_names = np.array(properties[[node_col]]) #array of edges to use as UIDs.
            indices = [tuple([node_name[0]]) for node_name in node_names]
            multi_index = pd.MultiIndex.from_tuples(indices, names=['nodes'])
            default_uid_cols = ['nodes']
            PS = create_df(properties,
                           uid_cols = [node_col],
                           default_uid_cols = default_uid_cols,
                           indices = indices, multi_index = multi_index,
                           weight_prop_col = node_weight_prop,
                           misc_prop_col = misc_properties,
                           default_weight = default_node_weight,
                           aggregation_methods = aggregation_methods)




    return PS


def to_property_store_from_dictionary():

    pass




# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for incidence stores
#-------------------------------------------------------------------------------------------------


def remove_incidence_store_duplicates(IS):
    return IS.drop_duplicates(keep='first', inplace = False)


def incidence_store_from_two_column_dataframe(setsystem, edge_col = 'edges', node_col = 'nodes'):
    '''
    Parameters
    ----------

    setsystem : (optional) dict of iterables, dict of dicts,iterable of iterables,
        pandas.DataFrame, numpy.ndarray, default = None
        See SetSystem above for additional setsystem requirements.

    edge_col : (optional) str | int, default = 0
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for (hyper)edge ids. Will be used to reference edgeids for
        all set systems.

    node_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for node ids. Will be used to reference nodeids for all set systems.

    '''


    #restructing set system to incidence store dataframe
    if len(setsystem) == 0: #if setsystem of length 0 or no set system is provided then return None for IS.
        IS = None
    else: #if non-empty setsystem
        #change column names to edges and nodes.
        column_renaming_dict = {edge_col: 'edges', node_col: 'nodes'}
        IS = setsystem.rename(column_renaming_dict)
        #remove duplicate rows with same ID
        IS = remove_incidence_store_duplicates(IS)

    return IS



# In[ ]: testing code
# Only runs if running from this file (This will show basic examples and testing of the code)


'''
to do:
    * program dictionary type properties
    * program other setsystem data types

'''


if __name__ == "__main__":
    incidence_dataframe = pd.DataFrame({'edges': ['a', 'a', 'a', 'b', 'c', 'c'], 'nodes': [1, 1, 2, 3, 2, 3]})

    cell_prop_dataframe = pd.DataFrame({'edges': ['a', 'a', 'a', 'b', 'c', 'c'], 'nodes': [1, 1, 2, 3, 2, 3],
                                       'color': ['red', 'red', 'red', 'red', 'red', 'blue'],
                                       'other_properties': [{}, {}, {}, {'time': 3}, {}, {}]})

    edge_prop_dataframe = pd.DataFrame({'edges': ['a', 'b', 'c'],
                                       'strength': [2, np.nan, 3]})

    node_prop_dataframe = pd.DataFrame({'nodes': [1],
                                       'temperature': [60]})



    print('Provided Dataframes')
    print('-'*100)
    display(incidence_dataframe)
    display(cell_prop_dataframe)
    display(edge_prop_dataframe)
    display(node_prop_dataframe)



    print('\n \nRestructured Dataframes using single factory method for property store repeated')
    print('-'*100)



    IS = incidence_store_from_two_column_dataframe(incidence_dataframe)
    display(IS)

    IPS = property_store_from_dataframe(properties = cell_prop_dataframe,
                                        property_type = 'cell_properties',
                                        misc_cell_properties = 'other_properties',
                                        aggregation_methods = {'weight': 'sum'},)
    display(IPS)

    EPS = property_store_from_dataframe(properties = edge_prop_dataframe,
                                        property_type = 'edge_properties',
                                        edge_weight_prop = 1)
    display(EPS)

    NPS = property_store_from_dataframe(properties = node_prop_dataframe,
                                        property_type = 'node_properties',)
    display(NPS)
