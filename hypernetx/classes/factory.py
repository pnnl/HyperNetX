
import pandas as pd
import networkx as nx
import numpy as np

# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for incidence stores
#-------------------------------------------------------------------------------------------------

def to_incidence_store_from_two_column_pandas_dataframe(pandas_dataframe, edge_col = 'edges', node_col = 'nodes'):
    #change column names to edges and nodes.
    column_renaming_dict = {edge_col: 'edges', node_col: 'nodes'}
    IS = pandas_dataframe.rename(column_renaming_dict)
    return IS





# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for property stores
#-------------------------------------------------------------------------------------------------

def to_property_store_from_dataframe(properties, property_type, 
                                     edge_col = 'edges', node_col = 'nodes', 
                                     misc_cell_properties = 'misc_properties', misc_properties = 'misc_properties',
                                     weight_prop = None, edge_weight_prop = 'weight', node_weight_prop = 'weight',
                                     cell_weight_col = 'weight',
                                     cell_weights = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0):
    
    
    
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
        
    """
    
    properties_df_dict = {}
    
    
    #create a multi-index with level and id columns.
    if property_type == 'cell_properties':
        incidence_pairs = np.array(properties[[edge_col, node_col]]) #array of incidence pairs to use as UIDs.
        indices = [tuple(incidence_pair) for incidence_pair in incidence_pairs]
        multi_index = pd.MultiIndex.from_tuples(indices, names=['edges', 'nodes'])
        # incidence_df = create_df(datadf, uid_col, weight_col, misc_col)
    elif property_type == 'edge_properties': 
        edge_names = np.array(properties[[edge_col]]) #array of edges to use as UIDs.
        indices = [tuple([edge_name[0]]) for edge_name in edge_names]
        multi_index = pd.MultiIndex.from_tuples(indices, names=['edges'])
        # edge_df = create_df(datadf, uid_col, weight_col, misc_col)
    elif property_type == 'node_properties': 
        node_names = np.array(properties[[node_col]]) #array of edges to use as UIDs.
        indices = [tuple([node_name[0]]) for node_name in node_names]
        multi_index = pd.MultiIndex.from_tuples(indices, names=['nodes'])
        

    #get names of property columns provided that are not the edge or node columns
    property_columns = set(list(properties.columns)) - set([edge_col, node_col])
    for prop in property_columns:
        properties_df_dict[prop] = properties.loc[:, prop]
        
        
    #get column names if integer was provided instead
    if isinstance(edge_col, int):
        edge_col = properties.columns[edge_col]
    if isinstance(edge_weight_prop, int):
        edge_weight_prop = properties.columns[edge_weight_prop]
    if isinstance(node_col, int):
        node_col = properties.columns[node_col]
    if isinstance(node_weight_prop, int):
        node_weight_prop = properties.columns[node_weight_prop]
    if isinstance(misc_properties, int):
        misc_properties = properties.columns[misc_properties]  
    if isinstance(cell_weight_col, int):
        cell_weight_col = properties.columns[cell_weight_col] 
    if isinstance(misc_properties, int):
        misc_cell_properties = properties.columns[misc_cell_properties] 
    
        
        
    #rename edge and node columns if needed to default names
    #change name of edges column to "edges"
    if edge_col != 'edges' and edge_col in properties_df_dict:
        # Pop the value associated with edges key and store it
        edge_column_values = properties_df_dict.pop(edge_col)
        # Add the popped value with the new edges name "edges"
        properties_df_dict['edges'] = edge_column_values
    #change name of nodes column to "nodes"
    if node_col != 'nodes' and node_col in properties_df_dict:
        # Pop the value associated with nodes key and store it
        node_column_values = properties_df_dict.pop(node_col)
        # Add the popped value with the new nodes name "nodes"
        properties_df_dict['nodes'] = node_column_values
        
        
        
    #if weight property is set then force all weight properties to be the weight property
    if weight_prop != None:
         edge_weight_prop = weight_prop
         node_weight_prop = weight_prop
         cell_weight_col = weight_prop
        
    
    if property_type == 'cell_properties':
        
        #check weight column exists or if weight col name exists in dictionary and assign it as column if it doesn't
        if cell_weight_col not in property_columns:
            create_default_weight_column = False
            if misc_cell_properties in property_columns:
                #make sure that an array of cells weights wasn't provided. use that if it was
                if not (isinstance(cell_weights, int) or isinstance(cell_weights, float)):
                    #check if cell_weight_col is a key in any of the misc properties dicitonary.
                    if any(cell_weight_col in misc_dict for misc_dict in properties_df_dict[misc_cell_properties]):
                        #create list of cell weights from misc properties dictionaries and use default value if not in keys
                        cell_weights_from_misc_dicts = []
                        for misc_dict in properties_df_dict[misc_cell_properties]:
                            if cell_weight_col in misc_dict:
                                cell_weights_from_misc_dicts.append(misc_dict[cell_weight_col])
                            else:
                                cell_weights_from_misc_dicts.append(cell_weights)
                        properties_df_dict[cell_weight_col] = cell_weights_from_misc_dicts
                    else:
                        create_default_weight_column = True
                else:
                    create_default_weight_column = True
            else:
                create_default_weight_column = True
                    
            if create_default_weight_column:
                if isinstance(cell_weights, int) or isinstance(cell_weights, float):
                    properties_df_dict[cell_weight_col] = [cell_weights]*len(indices)
                else:
                    properties_df_dict[cell_weight_col] = cell_weights
                    
        #change name of weight props column to "weight"
        if cell_weight_col != 'weight':
            # Pop the value associated with properties key and store it
            column_values = properties_df_dict.pop(cell_weight_col)
            # Add the popped value with the new properties name "properties"
            properties_df_dict['weight'] = column_values
            
        
        #set misc properties column if not already provided
        if misc_cell_properties not in property_columns:
            properties_df_dict[misc_cell_properties] = [{}]*len(indices)
        #change name of misc props column to "properties"
        if misc_cell_properties != 'misc_properties':
            # Pop the value associated with properties key and store it
            misc_props = properties_df_dict.pop(misc_cell_properties)
            # Add the popped value with the new properties name "properties"
            properties_df_dict['misc_properties'] = misc_props
   
    
    if property_type == 'edge_properties':
        #check weight column exists or if weight col name exists in dictionary and assign it as column if it doesn't
        if edge_weight_prop not in property_columns:
            create_default_weight_column = False
            if misc_properties in property_columns:
                #make sure that an array of cells weights wasn't provided. use that if it was
                if not (isinstance(default_edge_weight, int) or isinstance(default_edge_weight, float)):
                    #check if weight_prop is a key in any of the misc properties dicitonary.
                    if any(edge_weight_prop in misc_dict for misc_dict in properties_df_dict[misc_properties]):
                        #create list of weights from misc properties dictionaries and use default value if not in keys
                        weights_from_misc_dicts = []
                        for misc_dict in properties_df_dict[misc_properties]:
                            if edge_weight_prop in misc_dict:
                                weights_from_misc_dicts.append(misc_dict[edge_weight_prop])
                            else:
                                weights_from_misc_dicts.append(default_edge_weight)
                        properties_df_dict[edge_weight_prop] = weights_from_misc_dicts
                    else:
                        create_default_weight_column = True
                else:
                    create_default_weight_column = True
            else:
                create_default_weight_column = True
                    
            if create_default_weight_column:
                if isinstance(default_edge_weight, int) or isinstance(default_edge_weight, float):
                    properties_df_dict[edge_weight_prop] = [default_edge_weight]*len(indices)
                else:
                    properties_df_dict[edge_weight_prop] = default_edge_weight
                    
                    
        #change name of weight props column to "weight"
        if edge_weight_prop != 'weight':
            # Pop the value associated with the key and store it
            column_values = properties_df_dict.pop(edge_weight_prop)
            # Add the popped value with the new name
            properties_df_dict['weight'] = column_values
            
                
    if property_type == 'node_properties':
        #check weight column exists or if weight col name exists in dictionary and assign it as column if it doesn't
        if node_weight_prop not in property_columns:
            create_default_weight_column = False
            if misc_properties in property_columns:
                #make sure that an array of cells weights wasn't provided. use that if it was
                if not (isinstance(default_node_weight, int) or isinstance(default_node_weight, float)):
                    #check if weight_prop is a key in any of the misc properties dicitonary.
                    if any(node_weight_prop in misc_dict for misc_dict in properties_df_dict[misc_properties]):
                        #create list of weights from misc properties dictionaries and use default value if not in keys
                        weights_from_misc_dicts = []
                        for misc_dict in properties_df_dict[misc_properties]:
                            if node_weight_prop in misc_dict:
                                weights_from_misc_dicts.append(misc_dict[node_weight_prop])
                            else:
                                weights_from_misc_dicts.append(default_node_weight)
                        properties_df_dict[node_weight_prop] = weights_from_misc_dicts
                    else:
                        create_default_weight_column = True
                else:
                    create_default_weight_column = True
            else:
                create_default_weight_column = True
                    
            if create_default_weight_column:
                if isinstance(default_node_weight, int) or isinstance(default_node_weight, float):
                    properties_df_dict[node_weight_prop] = [default_node_weight]*len(indices)
                else:
                    properties_df_dict[node_weight_prop] = default_node_weight 
        
        
    if property_type == 'edge_properties' or property_type == 'node_properties':
        #set column if not already provided
        if misc_properties not in property_columns:
            properties_df_dict[misc_properties] = [{}]*len(indices)
        #change name of misc props column to "properties"
        if misc_properties != 'misc_properties':
            # Pop the value associated with properties key and store it
            misc_props = properties_df_dict.pop(misc_properties)
            # Add the popped value with the new properties name "properties"
            properties_df_dict['misc_properties'] = misc_props
    
    
    #create dataframe from property store dictionary object
    PS = pd.DataFrame(properties_df_dict)    
    #set multi index for dataframe
    PS = PS.set_index(multi_index)
    
    #remove any NaN values or missing values in weight column
    if property_type == 'node_properties':
        PS['weight'].fillna(default_node_weight, inplace = True)
    if property_type == 'edge_properties':
        PS['weight'].fillna(default_edge_weight, inplace = True)
    if property_type == 'cell_properties':
        PS['weight'].fillna(cell_weights, inplace = True)
    
    #reorder columns to have properties last 
    # Get the column names and the specific column
    column_names = list(PS.columns)
    specific_col = 'misc_properties'
    # Create a new order for the columns
    new_order = [col for col in column_names if col != specific_col] + [specific_col]
    # Reorder the dataframe using reindex
    PS = PS.reindex(columns=new_order)
    
    return PS

def to_property_store_from_dictionary():
    
    pass



# In[ ]:
#-------------------------------------------------------------------------------------------------
#type checks for factory methods
#-------------------------------------------------------------------------------------------------

def check_setsystem_data_type(data):
    data_type = 'two_column_dataframe'
    return data_type

def check_properties_data_type(data):
    data_type = 'dataframe'
    return data_type


# In[ ]:
#-------------------------------------------------------------------------------------------------
#main factory method helper functions
#-------------------------------------------------------------------------------------------------
def remove_property_store_duplicates(PS, property_type, aggregation_methods = {}):
    for col in PS.columns:
        if col not in aggregation_methods:
            aggregation_methods[col] = 'first'
    if property_type == 'cell_properties':
        return PS.groupby(level = ['edges', 'nodes']).agg(aggregation_methods)
    elif property_type == 'node_properties':
        return PS.groupby(level = ['nodes']).agg(aggregation_methods)
    elif property_type == 'edge_properties':
        return PS.groupby(level = ['edges']).agg(aggregation_methods)
    
def remove_incidence_store_duplicates(IS):
    return IS.drop_duplicates(keep='first', inplace = False)


# In[ ]:
#-------------------------------------------------------------------------------------------------
#incidence store factory methods
#-------------------------------------------------------------------------------------------------

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
        #get incidence store for the given data type.
        IS = to_incidence_store_from_two_column_pandas_dataframe(setsystem, edge_col, node_col)
        #remove duplicate rows with same ID
        IS = remove_incidence_store_duplicates(IS)


# In[ ]:
#-------------------------------------------------------------------------------------------------
#propertry store factory methods
#-------------------------------------------------------------------------------------------------

def incidence_property_store_from_dataframe(cell_properties = None, cell_aggregation_methods = {}, 
                     edge_col = 'edges', node_col = 'nodes', misc_cell_properties = 'misc_properties', 
                     cell_weight_col = 'weight',cell_weights = 1.0,):
    '''
    Parameters
    ----------


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

    cell_properties : (optional) Sequence[int | str] | Mapping[T,Mapping[T,Mapping[str,Any]]],
        default = None
        Column names from pd.DataFrame to use as cell properties
        or a dict assigning cell_property to incidence pairs of edges and
        nodes. Will generate a misc_cell_properties, which may have variable lengths per cell.

    misc_cell_properties : (optional) str | int, default = None
        Column name of dataframe corresponding to a column of variable
        length property dictionaries for the cell. Ignored for other setsystem
        types.
        
    cell_aggregation_methods : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    edge_aggregation_methods : (optional) dict, default = {}
        By default duplicate edges will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    node_aggregation_methods : (optional) dict, default = {}
        By default duplicate nodes will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    '''
    
        
    property_type = 'cell_properties'
    if cell_properties is None: #if no properties are provided for that property type.
        multi_index = pd.MultiIndex.from_tuples(levels=[[],[]], codes=[[],[]], names=['edges', 'nodes'])
        IPS = pd.DataFrame(index = multi_index, columns=['weight', 'properties'])
        
    else: #if properties are provided for that property type.
        IPS = to_property_store_from_dataframe(properties = cell_properties, property_type = property_type, 
                                              edge_col = edge_col, node_col = node_col, 
                                              misc_cell_properties = misc_cell_properties,
                                              cell_weight_col = cell_weight_col,
                                              cell_weights = cell_weights)
        # remove any duplicate indices and combine using aggregation methods (defaults to 'first' if none provided).
        IPS = remove_property_store_duplicates(IPS, property_type, aggregation_methods = cell_aggregation_methods)
    
    return IPS

def node_property_store_from_dataframe(node_properties = None, node_aggregation_methods = {},
                                       node_col = 'nodes', misc_properties = 'misc_properties', 
                                       node_weight_prop = 'weight', default_node_weight = 1.0):
    
    
    '''
    Parameters
    ----------

    node_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for node ids. Will be used to reference nodeids for all set systems.


    node_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with node ids.
        First column of dataframe or keys of dict link to node ids in
        setsystem.

    misc_properties : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    node_weight_prop : (optional) str, default = None,
        Name of property in node_properties to use for weight.

    default_node_weight : (optional) int | float, default = 1
        Used when node weight property is missing or undefined
        
    node_aggregation_methods : (optional) dict, default = {}
        By default duplicate nodes will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    '''
    
        
    property_type = 'node_properties'
    if node_properties is None: #if no properties are provided for that property type.
        multi_index = pd.MultiIndex.from_tuples(levels=[[]], codes=[[]], names=['nodes'])
        NPS = pd.DataFrame(index = multi_index, columns=['weight', 'properties'])
        
    else: #if properties are provided for that property type.
        NPS = to_property_store_from_dataframe(properties = node_properties, property_type = property_type, 
                                               node_col = node_col, 
                                               misc_properties = misc_properties,
                                               node_weight_prop = node_weight_prop,
                                               default_node_weight = default_node_weight)
        # remove any duplicate indices and combine using aggregation methods (defaults to 'first' if none provided).
        NPS = remove_property_store_duplicates(NPS, property_type, aggregation_methods = node_aggregation_methods)
    
    return NPS

def edge_property_store_from_dataframe(edge_properties = None, edge_aggregation_methods = {}, 
                                       edge_col = 'edges', misc_properties = 'misc_properties', 
                                       edge_weight_prop = 'weight', default_edge_weight = 1.0):
    
    
    '''
    Parameters
    ----------

    edge_col : (optional) str | int, default = 1
        column index (or name) in pandas.dataframe or numpy.ndarray,
        used for node ids. Will be used to reference nodeids for all set systems.


    edge_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with edge ids.
        First column of dataframe or keys of dict link to edge ids in
        setsystem.

    misc_properties : (optional) int | str, default = None
        Column of property dataframes with dtype=dict. Intended for variable
        length property dictionaries for the objects.

    edge_weight_prop : (optional) str, default = None,
        Name of property in edge_properties to use for weight.

    default_edge_weight : (optional) int | float, default = 1
        Used when edge weight property is missing or undefined.
        
    edge_aggregation_methods : (optional) dict, default = {}
        By default duplicate edges will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    '''
    
        
    property_type = 'edge_properties'
    if edge_properties is None: #if no properties are provided for that property type.
        multi_index = pd.MultiIndex.from_tuples(levels=[[]], codes=[[]], names=['edges'])
        EPS = pd.DataFrame(index = multi_index, columns=['weight', 'properties'])
        
    else: #if properties are provided for that property type.
        EPS = to_property_store_from_dataframe(properties = edge_properties, property_type = property_type, 
                                               edge_col = edge_col, 
                                               misc_properties = misc_properties,
                                               edge_weight_prop = edge_weight_prop,
                                               default_edge_weight = default_edge_weight)
        # remove any duplicate indices and combine using aggregation methods (defaults to 'first' if none provided).
        EPS = remove_property_store_duplicates(EPS, property_type, aggregation_methods = edge_aggregation_methods)
    
    return EPS



# In[ ]:
#-------------------------------------------------------------------------------------------------
#combined factory method (no longer used but keeping for now)
#-------------------------------------------------------------------------------------------------

def restructure_data(setsystem, setsystem_data_type = None,
                     cell_properties = None, edge_properties = None, node_properties = None, properties_data_type = None, 
                     
                     cell_aggregation_methods = {}, 
                     edge_aggregation_methods = {}, 
                     node_aggregation_methods = {},
                     
                     edge_col = 'edges', node_col = 'nodes', misc_properties = 'misc_properties',                                
                     misc_cell_properties = 'properties',         
                     
                     weight_prop = None, edge_weight_prop = 'weight', node_weight_prop = 'weight',
                     cell_weight_col = 'weight',
                     cell_weights = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0
                    ):
    
    

    
    
    
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

    cell_properties : (optional) Sequence[int | str] | Mapping[T,Mapping[T,Mapping[str,Any]]],
        default = None
        Column names from pd.DataFrame to use as cell properties
        or a dict assigning cell_property to incidence pairs of edges and
        nodes. Will generate a misc_cell_properties, which may have variable lengths per cell.

    misc_cell_properties : (optional) str | int, default = None
        Column name of dataframe corresponding to a column of variable
        length property dictionaries for the cell. Ignored for other setsystem
        types.

    edge_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with edge ids.
        First column of dataframe or keys of dict link to edge ids in
        setsystem.

    node_properties : (optional) pd.DataFrame | dict, default = None
        Properties associated with node ids.
        First column of dataframe or keys of dict link to node ids in
        setsystem.

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
        
    cell_aggregation_methods : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    edge_aggregation_methods : (optional) dict, default = {}
        By default duplicate edges will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    node_aggregation_methods : (optional) dict, default = {}
        By default duplicate nodes will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
        
        
        
        
    -----------------THINGS REMOVED OR CHANGED OR ADDED----------------
    
    REMOVED - I don't think this is necessary and could be added but is confusing if anything
    properties : (optional) pd.DataFrame | dict, default = None
        Concatenation/union of edge_properties and node_properties.
        By default, the object id is used and should be the first column of
        the dataframe, or key in the dict. If there are nodes and edges
        with the same ids and different properties then use the edge_properties
        and node_properties keywords.
        
        
        
        
        
    aggregateby : (optional) str, dict, default = 'first'
        By default duplicate edge,node incidences will be dropped unless
        specified with `aggregateby`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information.
    CHANGED TO
    cell_aggregation_methods : (optional) dict, default = {}
        By default duplicate incidences will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    edge_aggregation_methods : (optional) dict, default = {}
        By default duplicate edges will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
    node_aggregation_methods : (optional) dict, default = {}
        By default duplicate nodes will be dropped unless
        specified with `aggregation_methods`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information. An example aggregation method is {'weight': 'sum'} to sum 
        the weights of the aggregated duplicate rows.
        
        
        
    '''
    
    #restructing set system to incidence store dataframe
    if len(setsystem) == 0: #if setsystem of length 0 or no set system is provided then return None for IS.
        IS = None
    else: #if non-empty setsystem
        # restructing incidence store data
        if setsystem_data_type == None:
            setsystem_data_type = check_setsystem_data_type(setsystem)
        
        #get incidence store for the given data type.
        
        if setsystem_data_type == 'two_column_dataframe':
            IS = to_incidence_store_from_two_column_pandas_dataframe(setsystem)
        elif setsystem_data_type == 'iterable_of_iterables':
            IS = None #need to program
        elif setsystem_data_type == 'bipartite':
            IS = None #need to program
        elif setsystem_data_type == 'dictionary_of_iterables':
            IS = None #need to program
        elif setsystem_data_type == 'dictionary_of_dictionaries':
            IS = None #need to program
        elif setsystem_data_type == 'dataframe_with_attributes':
            IS = None #need to program
        else:
            raise ValueError("Provided setsystem_data_type is not a valid option. See documentation for available options.")
            
        IS = remove_incidence_store_duplicates(IS)
        
    #restructing properties data
    restructued_properties_dataframes = {}
    to_agg_methods = {'cell_properties': cell_aggregation_methods, 
                      'edge_properties': edge_aggregation_methods, 
                      'node_properties': node_aggregation_methods}
    
    for property_type, properties in {'cell_properties': cell_properties, 
                                      'edge_properties': edge_properties, 
                                      'node_properties': node_properties}.items():
        
        if properties is None: #if no properties are provided for that property type.
            PS = pd.DataFrame(columns=['uid', 'weight', 'properties'])
            
        else: #if properties are provided for that property type.
            if properties_data_type == None:
                properties_data_type = check_properties_data_type(properties)
                
            # get property store for the given data type.
            
            if properties_data_type == 'dataframe': 
                PS = to_property_store_from_dataframe(properties, property_type, edge_col, node_col, 
                                                      misc_cell_properties, misc_properties,
                                                      weight_prop, edge_weight_prop, node_weight_prop,
                                                      cell_weight_col,cell_weights, default_edge_weight, default_node_weight)
            elif properties_data_type == 'dictionary': 
                PS = None # need to program
            else:
                raise ValueError("Provided properties_data_type is not a valid option. See documentation for available options.")
        
            
            agg_methods = to_agg_methods[property_type]
            PS = remove_property_store_duplicates(PS, property_type, agg_methods)
            
        restructued_properties_dataframes[property_type] = PS
        
    IPS = restructued_properties_dataframes['cell_properties']
    EPS = restructued_properties_dataframes['edge_properties']
    NPS = restructued_properties_dataframes['node_properties']
    
    return IS, IPS, EPS, NPS


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
    
    
    
    print('\n \nRestructured Dataframes using individual factory methods')
    print('-'*100)
    
    IS = incidence_store_from_two_column_dataframe(incidence_dataframe)
    display(IS)
    
    IPS = incidence_property_store_from_dataframe(cell_prop_dataframe, misc_cell_properties = 'other_properties',
                                                  cell_aggregation_methods = {'weight': 'sum'},)
    display(IPS)
    
    EPS = edge_property_store_from_dataframe(edge_prop_dataframe, edge_weight_prop = 1)
    display(EPS)
    
    NPS = node_property_store_from_dataframe(node_prop_dataframe)
    display(NPS)
    



    print('\n \n (OLD METHOD) Restructured Dataframes using resctructure_data function')
    print('-'*100)
    IS, IPS, EPS, NPS = restructure_data(setsystem = incidence_dataframe, 
                                         setsystem_data_type = 'two_column_dataframe', 
                                         
                                         cell_properties = cell_prop_dataframe, 
                                         edge_properties = edge_prop_dataframe, 
                                         node_properties = node_prop_dataframe, 
                                         properties_data_type = 'dataframe', 
                                         
                                         misc_cell_properties = 'other_properties', 
                                         edge_weight_prop = 1,
                                         cell_aggregation_methods = {'weight': 'sum'})
    
    display(IS)
    display(IPS)
    display(EPS)
    display(NPS)
    
    
    
    
    
    
    
    
    
    
    