
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

def to_incidence_store_from_iterable_of_iterables(iterable_of_iterables):
    edge_list = []# edge component of incidence pairs as list
    node_list = []# node component of incidence pairs as list
    for edge, nodes_of_edge in enumerate(iterable_of_iterables):
        for node in nodes_of_edge:
            #append incidence pair of (edge, node) to edge and node lists, respectively.
            edge_list += [edge]
            node_list += [node]
    incidence_store_dict = {'edges': edge_list, 
                            'nodes': node_list}
    #Incidence Store (IS)
    IS = pd.DataFrame(incidence_store_dict)
    return IS

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
                                     edge_col = 'edges', node_col = 'nodes', weight_col = 'weight', misc_properties_col = 'properties',
                                     default_cell_weight = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0):
    properties_df_dict = {}
    
    #set UID column
    if property_type == 'cell_properties':
        incidence_pairs = np.array(properties[[edge_col, node_col]]) #array of incidence pairs to use as UIDs.
        properties_df_dict['uid'] = [tuple(incidence_pair) for incidence_pair in incidence_pairs]
    elif property_type == 'edge_properties': 
        edge_names = np.array(properties[[edge_col]]) #array of edges to use as UIDs.
        properties_df_dict['uid'] = [tuple(edge_name) for edge_name in edge_names]
    elif property_type == 'node_properties': 
        node_names = np.array(properties[[node_col]]) #array of edges to use as UIDs.
        properties_df_dict['uid'] = [tuple(node_name) for node_name in node_names]
    
    
    
    #set property columns including weight and misc props.
    
    #get names of property columns provided that are not the edge or node columns
    property_columns = set(list(properties.columns)) - set([edge_col, node_col])
    for prop in property_columns:
        properties_df_dict[prop] = properties.loc[:, prop]
        
    #set weight property column if not already provided
    default_weights = {'cell_properties': default_cell_weight, 'edge_properties': default_edge_weight, 'node_properties': default_node_weight}
    if weight_col not in property_columns:
        properties_df_dict[weight_col] = [default_weights[property_type]]*len(properties_df_dict['uid'])
    
    #set misc properties column if not already provided
    if misc_properties_col not in property_columns:
        properties_df_dict[misc_properties_col] = [{}]*len(properties_df_dict['uid'])

    return pd.DataFrame(properties_df_dict)

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


def restructure_data(setsystem, cell_properties = None, edge_properties = None, node_properties = None, 
                     setsystem_data_type = None, properties_data_type = None,
                     edge_col = 'edges', node_col = 'nodes', weight_col = 'weight', misc_properties_col = 'properties',
                     default_cell_weight = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0):
    '''
    notes:
        * need to add functionality for if the setsystem is empty or is none.
    '''
    
    # restructing incidence store data
    if setsystem_data_type == None:
        setsystem_data_type = check_setsystem_data_type(setsystem)
    
    #get incidence store for the given data type.
    if setsystem_data_type == 'iterable_of_iterables':
        IS = to_incidence_store_from_iterable_of_iterables(setsystem)
    elif setsystem_data_type == 'dataframe':
        IS = to_incidence_store_from_two_column_pandas_dataframe(setsystem)
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
        
        
    #restructing cell properties data
    all_properties = {'cell_properties': cell_properties, 'edge_properties': edge_properties, 'node_properties': node_properties}
    restructued_properties_dataframes = {}
    for property_type in all_properties:
        #get properties of type (e.g., cells).
        properties = all_properties[property_type]
        
        
        if properties is None: #if no properties are provided for that property type.
            df = pd.DataFrame(columns=['uid', 'weight', 'properties'])
            
        else: #if properties are provided for that property type.
            if properties_data_type == None:
                properties_data_type = check_properties_data_type(properties)
                
            # get property store for the given data type.
            if properties_data_type == 'dictionary': 
                df = to_property_store_from_dictionary(properties, property_type, 
                                                       edge_col = 'edges', node_col = 'nodes', weight_col = 'weight', misc_properties_col = 'properties',
                                                       default_cell_weight = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0)
            elif properties_data_type == 'dataframe': 
                df = to_property_store_from_dataframe(properties, property_type, 
                                                      edge_col = 'edges', node_col = 'nodes', weight_col = 'weight', misc_properties_col = 'properties',
                                                      default_cell_weight = 1.0, default_edge_weight = 1.0, default_node_weight = 1.0)
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
                                       'properties': [{}, {}, {'time': 3}, {}, {}]})
    
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
    IS, IPS, EPS, NPS = restructure_data(incidence_dataframe, cell_prop_dataframe, edge_prop_dataframe, node_prop_dataframe, 
                                         setsystem_data_type = 'dataframe', properties_data_type = 'dataframe', )
    display(IS)
    display(IPS)
    display(EPS)
    display(NPS)
    
    
    