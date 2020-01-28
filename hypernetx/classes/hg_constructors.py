# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import warnings
import numpy as np

from hypernetx.classes.entity import Entity, EntitySet
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.exception import HyperNetXError


def from_bipartite(B,set_names=[0,1],name=None):
	"""
	Static method creates a Hypergraph from a bipartite graph.

	Parameters
	----------

	B: nx.Graph()
		A networkx bipartite graph. Each node in the graph has a property
		'bipartite' taking one of two values

	set_names: iterable
		An ordered list :math:`[x_0, x_1]`, corresponding to the values 
		assigned to the bipartite property in B.

	name: hashable

	Returns
	-------
	: Hypergraph

	Notes
	-----
	A partition for the nodes in a bipartite graph generates a hypergraph as follows.
	For each node n in B with bipartite property equal to set_names[0] there is a 
	node n in the hypergraph.  For each node e in B with bipartite property
	equal to set_names[1], there is an edge in the hypergraph. 
	For each edge (n,e) in B add n to the edge e in the hypergraph.

	""" 
	import networkx as nx
	from networkx.algorithms import bipartite

	if not bipartite.is_bipartite(B):
		raise HyperNetxError('Error: Method requires a bipartite graph.')
	entities = []
	for n,d in B.nodes(data=True):
	    if d['bipartite'] == set_names[1]:
	        elements = []
	        for nei in B.neighbors(n):
	            elements.append(Entity(nei,[],properties=B.nodes(data=True)[nei]))
	        if elements:
	            entities.append(Entity(n,elements,properties=d))
	name = name or '_'
	return Hypergraph(EntitySet(name,entities),name=name)


def from_numpy_array(M,node_names=None, edge_names=None, name=None):
	"""
	Create a hypergraph from a real valued matrix represented as a numpy array with dimensions 2x2 

	Parameters
	----------
	M : real valued array-like object, dimensions=2x2
	    representing a real valued matrix with rows corresponding to nodes and columns to edges

	node_names : object, array-like, default=None
	    List of node names must be the same length as M.shape[0]. 
	    If None then the node names correspond to row indices with 'v' prepended.

	edge_names : object, array-like, default=None
	    List of edge names must have the same length as M.shape[1]. 
	    If None then the edge names correspond to column indices with 'e' prepended.

	name : hashable
	    
	Returns
	-------
	 : Hypergraph


	Note
	----
	The constructor does not generate empty edges. 
	All zero columns in M are removed and the names corresponding to these
	edges are discarded.

	"""  	    
	## Create names for nodes and edges
	## Validate the size of the node and edge arrays

	M = np.array(M)
	if len(M.shape) != (2):
		raise HyperNetXError('Input requires a 2 dimensional numpy array')

	if node_names is not None:
	    nodenames = np.array(node_names)
	    if len(nodenames) != M.shape[0]:
	        print('Number of node names does not match number of rows.')
	        return None
	else:
	    nodenames = np.array([f'v{idx}' for idx in range(M.shape[0])])

	if edge_names is not None:
	    edgenames = np.array(edge_names)
	    if len(edgenames) != M.shape[1]:
	        print('Number of edge names does not match number of columns.')
	else:
	    edgenames = np.array([f'e{jdx}' for jdx in range(M.shape[1])])

	## Remove empty column indices from M columns and edgenames
	colidx = np.array([jdx for jdx in range(M.shape[1]) if any(M[:,jdx])])
	colidxsum = np.sum(colidx)
	if not colidxsum:
		return Hypergraph()
	M = M[:,colidx]
	edgenames = edgenames[colidx]

	edict = dict()
	## Create an EntitySet of edges from M         
	for jdx,e in enumerate(edgenames):
	    edict[e] = nodenames[[idx for idx in range(M.shape[0]) if M[idx,jdx]]]
	            
	return Hypergraph(edict,name=name)


def from_dataframe(df, fillna=0, transpose=False, name=None):
    '''
    Create a hypergraph from a Pandas Dataframe object using index to label vertices
    and Columns to label edges. 

    Parameters
    ----------
    df : Pandas.Dataframe
        a real valued dataframe with a single index
    fillna : float, default = 0
        a real value to place in empty cell, all-zero columns will not generate
        an edge
    transpose : bool, default = 0
        option to transpose the dataframe, in this case df.Index will label the edges
        and df.columns will label the nodes
        
    Returns
    -------
    : Hypergraph

    Note
    ----
    The constructor does not generate empty edges. 
    All-zero columns in df are removed and the names corresponding to these
    edges are discarded.    
    '''
    import pandas as pd

    if type(df) != pd.core.frame.DataFrame:
        raise HyperNetXError('Error: Input object must be a pandas dataframe.') 
    if transpose:
        df = df.transpose()   
    node_names = np.array(df.index)
    edge_names = np.array(df.columns)
    M = df.fillna(fillna).values
    return from_numpy_array(M,node_names=node_names,edge_names=edge_names,name=name)


def read_csv(fpath, index_col=0, fillna=0, transpose=False, name=None, **kwargs):
    '''
    Wrapper for pandas from_csv method to generate a hypergraph:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Parameters
    ----------
    fpath : str, path object, or file-like object
        Any valid string path is acceptable including a URL.
        see pandas.read-csv for additional documentation
    index_col : int, sequence or bool, default = 0
        required for node labels.
        see Pandas.from_csv for additional documentation
    fillna : float, default = 0
        a real value to place in empty cell, all-zero columns will not generate
        an edge
    transpose : bool, default = False
        option to transpose the dataframe, in this case df.Index will label the edges
        and df.columns will label the nodes
    name : str
    	name of hypergraph to be constructed
    kwargs : keyword arguments to pass to pandas.from_csv method
        
    Returns
    -------
    : Hypergraph

    Notes
    -----
    Defaults assume that the csv file has node names in the first column and edge names in the 
    first (header) row. To change the behavior use key word arguments to pass to pandas method.
    '''
    import pandas as pd

    df = pd.read_csv(fpath, index_col=0, skip_blank_lines=True, name=name, **kwargs)
    return from_dataframe(df, fillna=fillna, transpose=transpose)

def to_dataframe(h):
	'''
	Generates pandas dataframe based on h.incidence_matrix with h.nodes.uidset 
	as index name and h.edges.uidset as column names.

	Parameters:
	-----------
	h : Hypergraph

	Returns:
	--------
	 : pandas.DataFrame
	'''
	import pandas as pd

	M,rowdict,coldict = h.edges.incidence_matrix(sparse=False, index=True)
	pdindex = [rowdict[idx] for idx in range(len(rowdict))]
	pdcolumns = [coldict[idx] for idx in range(len(coldict))]
	return pd.DataFrame(M,index=pdindex,columns=pdcolumns)
