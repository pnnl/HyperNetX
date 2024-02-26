
import pandas as pd
import networkx as nx
import numpy as np

'''
Thoughts:
    Why do we not just follow networkx as an example and ditch all of the factory methods 
    and only be able to add nodes and edges and incidences in a specific format? We would still have everything but it would be empty to start.
'''

# In[ ]:
#-------------------------------------------------------------------------------------------------
# Individual factory methods for incidence stores
#-------------------------------------------------------------------------------------------------
'''notes: 
   * worried that choosing edge names as integers is a problem if they match node names. This is the case for iterable of iterables. Assuming this is fine.
   * for IoI or any method without attributes do I return none or trivial property stores? Could in the future plan for 
     "simple" hypergraph that does not have properties. For now I am returning trivial property store.s (i.e., no properties except weight = 1)
     have default method that returns weight or properties if there are no properties provided. 
     Should still instantiate an empty dataframe with the correct columns though.
   * should the attributes on the repeated instances also be combined? Not sure exactly how this would be done. I'm leaning towards just doing weight for now and then
     maybe adding a lossless additional combined attributes later on if it is needed.'
     
   * can have a properties input into the hypergraph creation for incidences, edges, and nodes. 
   Also, incidence properties can be retrieved directly from the set system, but only for the incidences.
'''

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
                                     misc_cell_properties = 'properties', misc_properties = 'properties',
                                     weight_prop = None, edge_weight_prop = 'weight', node_weight_prop = 'weight',
                                     cell_weight_col = 'weight',
                                     cell_weights = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0):
    
    
    
    properties_df_dict = {}
    
    
    
    
    #create a multi-index with level and id columns.
    if property_type == 'cell_properties':
        incidence_pairs = np.array(properties[[edge_col, node_col]]) #array of incidence pairs to use as UIDs.
        indices = [tuple(incidence_pair) for incidence_pair in incidence_pairs]
    elif property_type == 'edge_properties': 
        edge_names = np.array(properties[[edge_col]]) #array of edges to use as UIDs.
        indices = [tuple(['e', edge_name[0]]) for edge_name in edge_names]
    elif property_type == 'node_properties': 
        node_names = np.array(properties[[node_col]]) #array of edges to use as UIDs.
        indices = [tuple(['n', node_name[0]]) for node_name in node_names]
    multi_index = pd.MultiIndex.from_tuples(indices, names=['level', 'id'])
    
    
    
    

    #get names of property columns provided that are not the edge or node columns
    property_columns = set(list(properties.columns)) - set([edge_col, node_col])
    for prop in property_columns:
        properties_df_dict[prop] = properties.loc[:, prop]
        
        
        
        
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
        
        #check weight column
        if cell_weight_col not in property_columns:
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
        if misc_cell_properties != 'properties':
            # Pop the value associated with properties key and store it
            misc_props = properties_df_dict.pop(misc_cell_properties)
            # Add the popped value with the new properties name "properties"
            properties_df_dict['properties'] = misc_props
   
    
    
    if property_type == 'edge_properties':
        #check weight column
        if edge_weight_prop not in property_columns:
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
        #check weight column
        if node_weight_prop not in property_columns:
            if isinstance(default_node_weight, int) or isinstance(default_node_weight, float):
                properties_df_dict[node_weight_prop] = [default_node_weight]*len(indices)
            else:
                properties_df_dict[node_weight_prop] = default_node_weight
        #change name of weight props column to "weight"
        if node_weight_prop != 'weight':
            # Pop the value associated with the key and store it
            column_values = properties_df_dict.pop(node_weight_prop)
            # Add the popped value with the new name
            properties_df_dict['weight'] = column_values    
        
        
        
        
        
    if property_type == 'edge_properties' or property_type == 'node_properties':
        #set column if not already provided
        if misc_properties not in property_columns:
            properties_df_dict[misc_properties] = [{}]*len(indices)
        #change name of misc props column to "properties"
        if misc_properties != 'properties':
            # Pop the value associated with properties key and store it
            misc_props = properties_df_dict.pop(misc_properties)
            # Add the popped value with the new properties name "properties"
            properties_df_dict['properties'] = misc_props
    
    
    

    PS = pd.DataFrame(properties_df_dict)    
    PS = PS.set_index(multi_index)
    
    
    
    #reorder columns to have properties last 
    # Get the column names and the specific column
    column_names = list(PS.columns)
    specific_col = 'properties'
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
    data_type = 'iterable_of_iterables'
    return data_type

def check_properties_data_type(data):
    data_type = 'iterable_of_iterables'
    return data_type

def combine_repeated_incidences(incidence_store_dataframe, incidence_property_store_dataframe, aggregateby):
    '''
    returns an updated version of the incidence store nd incidence property store where repeated incidences are grouped and the weight 
    in the incidence property store dataframe is updated based on the number of grouped incidences. 
    Also I need to include which method of keeping is used (e.g., 'first'). 
    
    This should be run within each of the factory methods after all stores are created.
    '''
    pass

def restructure_data(setsystem, setsystem_data_type = None,
                     cell_properties = None, edge_properties = None, node_properties = None, properties_data_type = None, 
                     aggregateby = 'first',
                     
                     edge_col = 'edges', node_col = 'nodes',                               
                     misc_cell_properties = 'properties', misc_properties = 'properties',          
                     weight_prop = None, edge_weight_prop = 'weight', node_weight_prop = 'weight',
                     cell_weight_col = 'weight',
                     cell_weights = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0
                    ):
    '''
    notes:
        * need to add functionality for if the setsystem is empty or is none.
        
        
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

    aggregateby : (optional) str, dict, default = 'first'
        By default duplicate edge,node incidences will be dropped unless
        specified with `aggregateby`.
        See pandas.DataFrame.agg() methods for additional syntax and usage
        information.

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
        
        
        
        
        
        
        
        
    -THINGS REMOVED-
    properties : (optional) pd.DataFrame | dict, default = None
        Concatenation/union of edge_properties and node_properties.
        By default, the object id is used and should be the first column of
        the dataframe, or key in the dict. If there are nodes and edges
        with the same ids and different properties then use the edge_properties
        and node_properties keywords.

    '''
    
    # restructing incidence store data
    if setsystem_data_type == None:
        setsystem_data_type = check_setsystem_data_type(setsystem)
    
    #get incidence store for the given data type.
    
    if setsystem_data_type == 'two_column_dataframe':
        IS = to_incidence_store_from_two_column_pandas_dataframe(setsystem)
    elif setsystem_data_type == 'iterable_of_iterables':
        IS = None
    elif setsystem_data_type == 'bipartite':
        IS = None
    elif setsystem_data_type == 'dictionary_of_iterables':
        IS = None
    elif setsystem_data_type == 'dictionary_of_dictionaries':
        IS = None
    elif setsystem_data_type == 'dataframe_with_attributes':
        IS = None
    else:
        raise ValueError("Provided setsystem_data_type is not a valid option. See documentation for available options.")
        
        
    #restructing properties data
    restructued_properties_dataframes = {}
    for property_type, properties in {'cell_properties': cell_properties, 
                                      'edge_properties': edge_properties, 
                                      'node_properties': node_properties}.items():
        
        if properties is None: #if no properties are provided for that property type.
            df = pd.DataFrame(columns=['uid', 'weight', 'properties'])
            
        else: #if properties are provided for that property type.
            if properties_data_type == None:
                properties_data_type = check_properties_data_type(properties)
                
            # get property store for the given data type.
            
            if properties_data_type == 'dataframe': 
                df = to_property_store_from_dataframe(properties, property_type, edge_col, node_col, 
                                                      misc_cell_properties, misc_properties,
                                                      weight_prop, edge_weight_prop, node_weight_prop,
                                                      cell_weight_col,cell_weights, default_edge_weight, default_node_weight)
            elif properties_data_type == 'dictionary': 
                df = None
            else:
                raise ValueError("Provided properties_data_type is not a valid option. See documentation for available options.")
        

        restructued_properties_dataframes[property_type] = df
        
    IPS = restructued_properties_dataframes['cell_properties']
    EPS = restructued_properties_dataframes['edge_properties']
    NPS = restructued_properties_dataframes['node_properties']
    
    return IS, IPS, EPS, NPS

# In[ ]: testing code
# Only runs if running from this file (This will show basic examples and testing of the code)
if __name__ == "__main__":
    incidence_dataframe = pd.DataFrame({'edges': ['a', 'a', 'b', 'c', 'c'], 'nodes': [1, 2, 3, 2, 3]})
    
    cell_prop_dataframe = pd.DataFrame({'edges': ['a', 'a', 'b', 'c', 'c'], 'nodes': [1, 2, 3, 2, 3], 
                                       'color': ['red', 'red', 'red', 'red', 'blue'], 
                                       'other_properties': [{}, {}, {'time': 3}, {}, {}]})
    
    edge_prop_dataframe = pd.DataFrame({'edges': ['a', 'b', 'c'], 
                                       'weight': [2, 3, 3]})
    
    node_prop_dataframe = pd.DataFrame({'nodes': [1], 
                                       'temperature': [60]})
    
    print('Provided Dataframes\n------------------------')
    display(incidence_dataframe)
    display(cell_prop_dataframe)
    display(edge_prop_dataframe)
    display(node_prop_dataframe)
    
    print('\n \nRestructured Dataframes\n------------------------')
    IS, IPS, EPS, NPS = restructure_data(setsystem = incidence_dataframe, setsystem_data_type = 'two_column_dataframe', 
                                         cell_properties = cell_prop_dataframe, 
                                         edge_properties = edge_prop_dataframe, 
                                         node_properties = node_prop_dataframe, 
                                         properties_data_type = 'dataframe', 
                                         misc_cell_properties = 'other_properties')
    display(IS)
    display(IPS)
    display(EPS)
    display(NPS)

    