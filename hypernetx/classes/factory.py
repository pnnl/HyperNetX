# make PS for nodes and edges use uid for index.
# fix bug

import pandas as pd
import numpy as np
from hypernetx.classes.helpers import dict_depth


# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for property stores
#-------------------------------------------------------------------------------------------------


def remove_property_store_duplicates(PS, default_uid_col_names, aggregation_methods = {}):
    agg_methods = {}
    for col in PS.columns:
        if col not in aggregation_methods:
            agg_methods[col] = 'first'
        else:
            agg_methods[col] = aggregation_methods[col]
    return PS.groupby(level = default_uid_col_names).agg(agg_methods)

# def createdf(dfp,
#              uid_cols = None,
#              level = 0,
#              use_index = False,
#              weight_prop = None,
#              default_weight = 1.0,
#              misc_properties_col = None,):
#     if not isinstance(properties,pd.DataFrame):
#         raise TypeError('method requires a Pandas DataFrame')
#     else:    
        # dfp = deepcopy(properties)   ### not sure if this is wise
#         dfp.drop_duplicates(inplace=True)
#         if use_index == False:
#             if uid_cols != None:
#                 chk = lambda c : c if isinstance(c,str) else dfp.columns[c]
#                 dfp = dfp.set_index([chk(c) for c in uid_cols])
#             else:
#                 if level == 2:
#                     dfp = dfp.set_index(dfp.columns[0],dfp.columns[1])
#                 else:
#                     dfp = dfp.set_index(dfp.columns[0])
            
#         if weight_prop is not None and weight_prop in dfp.columns:
#             dfp = dfp.rename(columns={weight_prop: 'weight'})
#             dfp = dfp.fillna(
#                 {'weight': default_weight}
#             ) 
#         else:
#             dfp['weight'] = default_weight 

#         if misc_properties_col in dfp.columns and misc_properties_col != 'misc_properties':
#             dfp = dfp.rename(columns={misc_properties_col: 'misc_properties'}) 
#             prop_flag = 1
#             dfp.misc_properties.fillna({})            
#         else:
#             dfp['misc_properties'] = [{} for row in dfp.index]
#             prop_flag = 0 

#     cols = [c for c in dfp.columns if c not in ['uid','weight','misc_properties'] ] 
#     dfp = dfp[['weight'] + cols + ['misc_properties']]   
#     if prop_flag == 0:
#         return dfp
#     else:
#         dfp.misc_properties = dfp.misc_properties.map(mkdict)
#         return dfp

def create_df(properties, uid_cols, indices, multi_index,
              default_uid_col_names, weight_prop_col,
              misc_prop_col, default_weight, aggregation_methods):

    # initialize a dictionary to be converted to a pandas dataframe.
    properties_df_dict = {}

    #get names of property columns provided that are not the edge or node columns
    property_columns = set(list(properties.columns)) - set(uid_cols)
    for prop in property_columns: #set those as rows in DF
        properties_df_dict[prop] = properties.loc[:, prop]
    
    #get column names if integer was provided instead
    if isinstance(weight_prop_col, int):
        weight_prop_col = properties.columns[weight_prop_col]
    if isinstance(misc_prop_col, int):
        misc_prop_col = properties.columns[misc_prop_col]


    #rename uid columns if needed to default names.
    for i in range(len(uid_cols)):
        col, default_col = uid_cols[i], default_uid_col_names[i]
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
        properties_df_dict[misc_prop_col] = [{}]*len(indices)  ### TODO indices = df.index after deduping.
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
    PS = remove_property_store_duplicates(PS, default_uid_col_names, aggregation_methods = aggregation_methods)

    #reorder columns to have properties last
    # Get the column names and the specific column
    column_names = list(PS.columns)
    specific_col = 'misc_properties'
    # Create a new order for the columns
    new_order = [col for col in column_names if col != specific_col] + [specific_col]
    # Reorder the dataframe using reindex
    PS = PS.reindex(columns=new_order)

    return PS

###### TODO maybe have a use_index=False kwarg indicating if the uids are already the index
###### TODO lets keep life simple - assume uid as the default column name or 'edges','nodes' if level == 2
def dataframe_factory_method(DF, level, 
                             uid_cols = None, default_uid_col_names = ['edges','nodes'],
                             misc_properties_col = 'misc_properties', 
                             weight_col = 'weight', 
                             default_weight = 1.0,
                             aggregate_by = {}):
    
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
        
    default_uid_col_names : (optional) list of str, default = None
        Name of columns for the edges, nodes, or (edge,node) pair columns 
        to be renamed to in the multiindex.
        If None, then uses uid_cols column names; which, if also None, uses the first and/or second column names.

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
    
#### TODO if DF is None then you can't create a dataframe. So return None.
    if DF is None: #if no properties are provided for that property type.
        #check if default_uid_col_names are provided. if not set them to edges and/or nodes based on level.
        if default_uid_col_names is None:
            if level == 2:
                default_uid_col_names = ['edges', 'nodes']
            elif level == 1 or level == 0:
                default_uid_col_names = ['uid']
                
        multi_index = pd.MultiIndex.from_tuples([], names=default_uid_col_names)

        PS = pd.DataFrame(index = multi_index, columns=['weight', 'misc_properties'])

    else:
        #uid column name setting if they are not provided
        if uid_cols == None: #if none are provided set to the names of the first or first two columns depending on level
            if level == 0 or level == 1:
                uid_cols = [DF.columns[0]]
            elif level == 2:
                uid_cols = [DF.columns[0], DF.columns[1]]
                
            
        
        #default uid column name setting if they are not provided.
        if default_uid_col_names is None:
            if level == 0 or level == 1:
                default_uid_col_names = ['uid']
            elif level == 2:
                default_uid_col_names = ['edges', 'nodes']
            
        #error checking on uid_cols length
        if len(uid_cols) != 1 and (level == 0 or level == 1):
            raise ValueError("For level 0 or 1, the uid_cols must be a list and have length of 1.")
        elif len(uid_cols) != 2 and level == 2:
            raise ValueError("For level 2, the uid_cols must be a list and have length of 2.")
            
        
        #get column names if integer was provided instead and create new uid_cols with string names.
        uid_cols_to_str = []
        for col in uid_cols:
            if isinstance(col, int):
                uid_cols_to_str.append(DF.columns[col])
            else:
                uid_cols_to_str.append(col)
        uid_cols = uid_cols_to_str
        
        uids = np.array(DF[uid_cols]) #array of incidence pairs to use as UIDs.
        indices = [tuple(uid) for uid in uids]
        # set multi index to be used in property store dataframe
        multi_index = pd.MultiIndex.from_tuples(indices, names=default_uid_col_names)
        #create property store dataframe
        #### TODO look at the
        PS = create_df(DF,
                       uid_cols = uid_cols,
                       default_uid_col_names = default_uid_col_names,
                       indices = indices, 
                       multi_index = multi_index,
                       weight_prop_col = weight_col,
                       misc_prop_col = misc_properties_col,
                       default_weight = default_weight,
                       aggregation_methods = aggregate_by)

    return PS



def dict_factory_method(D, level, 
                        uid_cols = None, default_uid_col_names = None,
                        misc_properties_col = 'misc_properties', 
                        weight_col = 'weight', 
                        default_weight = 1.0,
                        aggregate_by = {}):
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
        
    default_uid_col_names : (optional) list of str, default = None
        Name of columns for the edges, nodes, or (edge,node) pair columns 
        to be renamed to in the multiindex.
        If None, then uses uid_cols column names; which, if also None, uses the first and/or second column names.

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
    ### Look up Series.explode:
    ###  d = {'a':[1,2,3],'b':[5,6,7]}
    ### pd.Series(d).explode()
    ### Returns:
        # a    1
        # a    2
        # a    3
        # b    5
        # b    6
        # b    7
        
    #if no dictionary is provided set it to an empty dictionary.
    if D is None:
        DF = None
    # if the dictionary data provided is for the setsystem (incidence data)
    elif level == 2:
        # get incidence pairs from dictionary keys and values.
        incidence_pairs = []
        for edge in D:
            for node in D[edge]:
                incidence_pairs.append([edge, node])
        DF = pd.DataFrame(incidence_pairs, columns = uid_cols)
        #if attributes are stored on the dictionary (ie, it has a depth greater than 2)
        if dict_depth(D) > 2:
            attribute_data = []
            for incidence_pair in incidence_pairs:
                edge, node = incidence_pair
                attributes_of_incidence_pair = D[edge][node]
                attribute_data.append(attributes_of_incidence_pair)
            attribute_df = pd.DataFrame(attribute_data)
            DF = pd.concat([DF, attribute_df], axis = 1)
    
    else:
        attribute_data = []
        for key in D:
            attributes_of_key = D[key]
            attribute_data.append(attributes_of_key)
        attribute_df = pd.DataFrame(attribute_data)
        DF = pd.concat([pd.DataFrame(list(D.keys()), columns = uid_cols), attribute_df], axis = 1)
        
        
    # get property store from dataframe
    PS = dataframe_factory_method(DF, level = level,
                                  uid_cols = uid_cols, default_uid_col_names = default_uid_col_names,
                                  misc_properties_col = misc_properties_col, 
                                  weight_col = weight_col, 
                                  default_weight = default_weight,
                                  aggregate_by = aggregate_by)
    return PS


def list_factory_method(L, level, 
                        uid_cols = None, default_uid_col_names = None,
                        misc_properties_col = 'misc_properties', 
                        weight_col = 'weight', 
                        default_weight = 1.0,
                        aggregate_by = {}):
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
        
    default_uid_col_names : (optional) list of str, default = None
        Name of columns for the edges, nodes, or (edge,node) pair columns 
        to be renamed to in the multiindex.
        If None, then uses uid_cols column names; which, if also None, uses the first and/or second column names.

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
    
    #explode list of lists into incidence pairs as a pandas dataframe using pandas series explode.
    DF = pd.DataFrame(pd.Series(L).explode()).reset_index()
    #rename columns to correct column names for edges and nodes using default_uid_col_names
    if default_uid_col_names is None: #if no uid_cols
        default_uid_col_names = ['edges', 'nodes']
    DF = DF.rename(columns=dict(zip(DF.columns, default_uid_col_names)))
    #create property store from dataframe.
    PS = dataframe_factory_method(DF, level= level,
                                  uid_cols = uid_cols, default_uid_col_names = default_uid_col_names,
                                  misc_properties_col = misc_properties_col, 
                                  weight_col = weight_col, 
                                  default_weight = default_weight,
                                  aggregate_by = aggregate_by)
    
    return PS

