

import pandas as pd
import networkx as nx
import numpy as np


# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for property stores
#-------------------------------------------------------------------------------------------------


def remove_property_store_duplicates(PS, default_uid_cols, aggregation_methods = {}):
    agg_methods = {}
    for col in PS.columns:
        if col not in aggregation_methods:
            agg_methods[col] = 'first'
        else:
            agg_methods[col] = aggregation_methods[col]
    return PS.groupby(level = default_uid_cols).agg(agg_methods)


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



def dataframe_factory_method(DF, uid_cols, default_uid_cols = None, 
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
    
    #error checking
    if len(uid_cols) != 1 and len(uid_cols) != 2:
        raise ValueError("uid_col must be a list and have length of 1 or 2.")
    
    #use provided column uids if none are specified
    if default_uid_cols == None:
        default_uid_cols = uid_cols
        
    if DF is None: #if no properties are provided for that property type.
        multi_index = pd.MultiIndex.from_tuples([], names=default_uid_cols)

        PS = pd.DataFrame(index = multi_index, columns=['weight', 'misc_properties'])

    else:
        uids = np.array(DF[uid_cols]) #array of incidence pairs to use as UIDs.
        indices = [tuple(uid) for uid in uids]
        multi_index = pd.MultiIndex.from_tuples(indices, names=default_uid_cols)
        PS = create_df(DF,
                       uid_cols = uid_cols,
                       default_uid_cols = default_uid_cols,
                       indices = indices, 
                       multi_index = multi_index,
                       weight_prop_col = weight_col,
                       misc_prop_col = misc_properties_col,
                       default_weight = default_weight,
                       aggregation_methods = aggregate_by)

    return PS






def dict_depth(dic, level = 0):
     
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1) for key in dic) 

def dict_factory_method(D, is_setsystem, default_uid_cols = None,
                        misc_properties_col = 'misc_properties', 
                        weight_col = 'weight',
                        default_weight = 1.0,
                        aggregate_by = {}):
    '''

    Parameters
    ----------
    D : dict
        DESCRIPTION.
    is_setsystem : bool
        Does the provided dictionary a setsystem (i.e., holds structure)?
        If false then the dictionary should be for edge or node properties.
    default_uid_cols : TYPE
        DESCRIPTION.
    misc_properties_col : TYPE
        DESCRIPTION.
    weight_col : TYPE
        DESCRIPTION.
    default_weight : TYPE
        DESCRIPTION.
    aggregate_by : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.

    '''
    
    
    #if no dictionary is provided set it to an empty dictionary.
    if D is None:
        D = {}
    
    
    if is_setsystem:
        #use provided column uids if specified
        if default_uid_cols != None:
            uid_cols = default_uid_cols
        else:
            uid_cols = ['edges', 'nodes']
            


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
            
        PS = dataframe_factory_method(DF, uid_cols = uid_cols, 
                                     misc_properties_col = misc_properties_col, 
                                     weight_col = weight_col, 
                                     default_weight = default_weight,
                                     aggregate_by = aggregate_by)
        return PS
    
    else:
        #use provided column uids if specified
        if default_uid_cols != None:
            uid_cols = default_uid_cols
        else:
            uid_cols = ['uid']

        attribute_data = []
        for key in D:
            attributes_of_key = D[key]
            attribute_data.append(attributes_of_key)
    
        attribute_df = pd.DataFrame(attribute_data)
        
        
            
        DF = pd.concat([pd.DataFrame(list(D.keys()), columns = uid_cols), attribute_df], axis = 1)
        PS = dataframe_factory_method(DF, uid_cols = uid_cols, 
                                     misc_properties_col = misc_properties_col, 
                                     weight_col = weight_col, 
                                     default_weight = default_weight,
                                     aggregate_by = aggregate_by)
        return PS


def list_factory_method(L, uid_cols = ['edges', 'nodes'],
                        misc_properties_col = 'misc_properties', 
                        weight_col = 'weight',
                        default_weight = 1.0,
                        aggregate_by = {}):
    '''

    Parameters
    ----------
    L : list
        DESCRIPTION.
    default_uid_cols : TYPE
        DESCRIPTION.
    misc_properties_col : TYPE
        DESCRIPTION.
    weight_col : TYPE
        DESCRIPTION.
    default_weight : TYPE
        DESCRIPTION.
    aggregate_by : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.
    '''
    
    # get incidence pairs from list of lists
    incidence_pairs = []
    for i, edge in enumerate(L):
        for node in edge:
            incidence_pairs.append([i, node])
    DF = pd.DataFrame(incidence_pairs, columns = uid_cols)
    PS = dataframe_factory_method(DF, uid_cols = uid_cols, 
                                 misc_properties_col = misc_properties_col, 
                                 weight_col = weight_col, 
                                 default_weight = default_weight,
                                 aggregate_by = aggregate_by)
    
    return PS

# In[ ]: testing code
# Only runs if running from this file (This will show basic examples and testing of the code)


'''
to do:
    * program dictionary type properties
    * program other setsystem data types
    
IPS = dataframe_factory_method(setsystem_df,index_cols=[edge_col,node_col],weight_col,default_weight,misc_properties,aggregate_by)

EPS = dataframe_factory_method(edge_properties_df,index_cols=[edge_uid_col],edge_weight_col,default_edge_weight,misc_edge_properties)
    
'''


if __name__ == "__main__":
    
    run_dataframe_example = False
    if run_dataframe_example:
        
        list_of_iterables = [[1, 1, 2], {1, 2}, {1, 2, 3}]
        display(list_of_iterables)
        
        IPS = list_factory_method(list_of_iterables, aggregate_by = {'weight': 'sum'})
        display(IPS)
        print('-'*100)
        
        
        
    run_simple_dict_example = False
    if run_simple_dict_example:
        
        cell_dict = {'e1':[1,2],'e2':[1,2],'e3':[1,2,3]}
    
        print('Provided Dataframes')
        print('-'*100)
        display(cell_dict)
        
        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)
        
        IPS = dict_factory_method(cell_dict, is_setsystem = True)
        
        display(IPS)
        print('-'*100)
    
        
    run_dict_example = False
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
        
        IPS = dict_factory_method(cell_prop_dict, is_setsystem = True, weight_col = 'w')
        display(IPS)
        
        
        EPS = dict_factory_method(edge_prop_dict, is_setsystem = False, default_uid_cols = ['edges'])
        display(EPS)
        
        
        NPS = dict_factory_method(None, is_setsystem = False, weight_col = 'w')
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
    
    
    
        IPS = dataframe_factory_method(incidence_dataframe, 
                                       uid_cols = ['e', 'n'], 
                                       default_uid_cols = ['edges', 'nodes'],
                                       aggregate_by = {'weight': 'sum'},)
        IS = IPS.index
        
        display(IS)
        display(IPS)
        
        NPS = dataframe_factory_method(None, uid_cols = ['nodes'])
        display(NPS)
        print('-'*100)
        
        
    run_dataframe_example = True
    if run_dataframe_example:
        
        cell_prop_dataframe = pd.DataFrame({'edges': ['a', 'a', 'a', 'b', 'c', 'c'], 'nodes': [1, 1, 2, 3, 2, 3],
                                            'color': ['red', 'red', 'red', 'red', 'red', 'blue'],
                                            'other_properties': [{}, {}, {}, {'time': 3}, {}, {}]})
    
        edge_prop_dataframe = pd.DataFrame({'edges': ['a', 'b', 'c'],
                                            'strength': [2, np.nan, 3]})
    
        node_prop_dataframe = pd.DataFrame({'nodes': [1],
                                            'temperature': [60]})
    
    
        print('Provided Dataframes')
        print('-'*100)
        display(cell_prop_dataframe)
        display(edge_prop_dataframe)
        display(node_prop_dataframe)
    
    
    
        print('\n \nRestructured Dataframes using single factory method for property store repeated')
        print('-'*100)
    
    
    
        IPS = dataframe_factory_method(cell_prop_dataframe, 
                                       uid_cols = ['edges', 'nodes'], 
                                       misc_properties_col = 'other_properties',
                                       aggregate_by = {'weight': 'sum'},)
        IS = IPS.index
        
        display(IS)
        
        display(IPS)
        
    
        EPS = dataframe_factory_method(edge_prop_dataframe, 
                                       uid_cols = ['edges'],
                                       weight_col = 1)
        display(EPS)
        
    
        NPS = dataframe_factory_method(node_prop_dataframe, 
                                       uid_cols = ['nodes'],)
        display(NPS)
        print('-'*100)
        
        from tqdm import tqdm
        for i in tqdm(range(10000)):
            IPS = dataframe_factory_method(cell_prop_dataframe, 
                                           uid_cols = ['edges', 'nodes'], 
                                           misc_properties_col = 'other_properties',
                                           aggregate_by = {'weight': 'sum'},)
    
    
    
