"""
This module contains methods which compute various distributions for hypergraphs:
    * Edge size distribution
    * Node degree distribution
    * Component size distribution
    * Toplex size distribution
    * Diameter

Also computes general hypergraph information: number of nodes, edges, cells, aspect ratio, incidence matrix density
"""

import numpy as np
import networkx as nx

from hypernetx import *

def centrality_stats(X):
    '''
    Computes basic centrality statistics for X

    Parameters
    ----------
    X : 
        an iterable of numbers

    Returns
    -------
    [min, max, mean, median, standard deviation] : list
        List of centrality statistics for X
    '''
    return [min(X), max(X), np.mean(X), np.median(X), np.std(X)]

def frequency_distribution(X): 
    '''
    Computes frequencies of elements of X.

    Parameters
    ----------
    X : iterable
        an iterable of numbers

    Returns
    -------
     frequency_distribution : dict
        Dictionary keyed on unique elements of X with values equal to their frequencies within X.
    '''
    x = np.unique(X,return_counts=True)
    return dict(zip(x[0],x[1]))

def edge_size_dist(H, aggregated=False):
    '''
    Computes edge sizes of a hypergraph.
    
    Parameters
    ----------
    H : Hypergraph
    aggregated : 
        If aggregated is True, returns a dictionary of 
        edge sizes and counts. If aggregated is False, returns a 
        list of edge sizes in H.
        
    Returns
    -------
     edge_size_dist : list or dict
        List of edge sizes or dictionary of edge size distribution.
    
    '''
    distr = [len(H.edges[e]) for e in H.edges]
    # distr = [len(e.uidset) for e in H.edges.elements.values()]
    if aggregated:
        return frequency_distribution(distr)
    else:
        return distr

def degree_dist(H, aggregated=False):
    '''
    Computes degrees of nodes of a hypergraph.
    
    Parameters
    ----------
    H : Hypergraph
    aggregated : 
        If aggregated is True, returns a dictionary of 
        degrees and counts. If aggregated is False, returns a 
        list of degrees in H.
        
    Returns
    -------
     degree_dist : list or dict
        List of degrees or dictionary of degree distribution
    '''
    distr = [H.degree(n) for n in H.nodes]
    if aggregated:
        return frequency_distribution(distr)
    else:
        return distr

def comp_dist(H, aggregated=False):
    '''
    Computes component sizes, number of nodes.
    
    Parameters
    ----------
    H : Hypergraph
    aggregated : 
        If aggregated is True, returns a dictionary of 
        component sizes (number of nodes) and counts. If aggregated 
        is False, returns a list of components sizes in H.
        
    Returns
    -------
     comp_dist : list or dictionary
        List of component sizes or dictionary of component size distribution
    
    See Also
    --------
    s_comp_dist
    
    '''
    distr = [len(c) for c in H.components()]
    if aggregated:
        return frequency_distribution(distr)
    else:
        return distr

def s_comp_dist(H, s=1, aggregated=False, edges=True):
    '''
    Computes s-component sizes, counting nodes or edges. 
    
    Parameters
    ----------
    H : Hypergraph
    s : positive integer, default is 1
    aggregated :
        If aggregated is True, returns a dictionary of
        s-component sizes and counts in H. If aggregated is 
        False, returns a list of s-component sizes in H.
    edges :
        If edges is True, the component size is number of edges.
        If edges is False, the component size is number of nodes.
    
    Returns
    -------
     s_comp_dist : list or dictionary
        List of component sizes or dictionary of component size distribution in H
    
    See Also
    --------
    comp_dist
    
    '''
    comps = list(H.s_component_subgraphs(s=s))
    if edges:
        distr = [len(c.edges) for c in comps]
    else:
        distr = [len(c.nodes) for c in comps]
    
    if aggregated:
        return frequency_distribution(distr)
    else:
        return distr

def toplex_dist(H, aggregated=False):
    '''
    
    Computes toplex sizes for hypergraph H.
    
    Parameters
    ----------
    H : Hypergraph
    aggregated : 
        If aggregated is True, returns a dictionary of 
        toplex sizes and counts in H. If aggregated 
        is False, returns a list of toplex sizes in H.
    
    Returns
    -------
     toplex_dist : list or dictionary
        List of toplex sizes or dictionary of toplex size distribution in H
    '''
    tops = H.toplexes()
    distr = [len(e) for e in tops.edges()]
    
    if aggregated:
        return frequency_distribution(distr)
    else:
        return distr

def s_node_diameter_dist(H):
    '''
    Parameters
    ----------
    H : Hypergraph
    
    Returns
    -------
     s_node_diameter_dist : list
        List of s-node-diameters for hypergraph H starting with s=1 
        and going up as long as the hypergraph is s-node-connected
    '''
    i=1
    diams = []
    while H.is_connected(s=i):
        diams.append(H.diameter(s=i))
        i += 1 
    return diams

def s_edge_diameter_dist(H):
    '''    
    Parameters
    ----------
    H : Hypergraph
    
    Returns
    -------
     s_edge_diameter_dist : list
        List of s-edge-diameters for hypergraph H starting with s=1 
        and going up as long as the hypergraph is s-edge-connected
    '''
    i=1
    diams = []
    while H.is_connected(s=i,edges=True):
        diams.append(H.edge_diameter(s=i))
        i += 1        
    return diams

def info(H,obj=None, dictionary=False):
    '''
    Print a summary of simple statistics for H

    Parameters
    ----------
    H : Hypergraph
    obj : optional
        either a node or edge uid from the hypergraph  
    dictionary : optional
        If True then returns the info as a dictionary rather 
        than a string
        If False (default) returns the info as a string

    Returns
    -------
     info : string
        Returns a string of statistics of the size, 
        aspect ratio, and density of the hypergraph. 
        Print the string to see it formatted.

    '''
    if dictionary:
        return info_dict(H, obj=obj)
    
    report = dict()
    if not H.edges.elements:
        return f'Hypergraph {H.name} is empty.'
    info = ""
    if obj:
        if obj in H.nodes:
            membs = list(H.nodes[obj].memberships)
            info += f'Node "{obj}" has the following properties:\n' 
            info += f'Degree: {len(membs)}\n'
            info += f'Contained in: {membs}\n'
            info += f'Neighbors: {list(H.neighbors(obj))}'
        if obj in H.edges:
            info += f'Edge "{obj}" has the following properties:\n'
            info += f'Size: {H.edges[obj].size()}\n'
            info += f'Elements: {list(H.edges[obj].uidset)}'
    else:
        lnodes = len(H.nodes)
        ledges = len(H.edges)
        M = H.incidence_matrix(index=False)
        ncells = M.nnz
        info += f'Number of Rows: {lnodes}\n'
        info += f'Number of Columns: {ledges}\n'
        info += f'Aspect Ratio: {lnodes/(ledges)}\n'
        info += f'Number of non-empty Cells: {ncells}\n'
        info += f'Density: {ncells/(lnodes*ledges)}'
    return info

def info_dict(H, obj=None):
    '''
    Create a summary of simple statistics for H

    Parameters
    ----------
    H : Hypergraph
    obj : optional
        either a node or edge uid from the hypergraph  

    Returns
    -------
     info_dict : dict
        Returns a dictionary of statistics of the size, 
        aspect ratio, and density of the hypergraph. 

    '''
    report = dict()
    if not H.edges.elements:
        return {}
    
    if obj:
        if obj in H.nodes:
            membs = list(H.nodes[obj].memberships)
            report['degree'] = len(membs)

        if obj in H.edges:
            report['size'] = H.edges[obj].size()

    else:
        lnodes = len(H.nodes)
        ledges = len(H.edges)
        M = H.incidence_matrix(index=False)
        ncells = M.nnz
        
        report['nrows'] = lnodes
        report['ncols'] = ledges
        report['aspect ratio'] = lnodes/ledges
        report['ncells'] = ncells
        report['density'] = ncells/(lnodes*ledges)

    return report

def dist_stats(H):
    """
    Computes many basic hypergraph stats and puts them all into a single dictionary object 
    
        * nrows = number of nodes (rows in the incidence matrix)
        * ncols = number of edges (columns in the incidence matrix)
        * aspect ratio = nrows/ncols
        * ncells = number of filled cells in incidence matrix
        * density = ncells/(nrows*ncols)
        * node degree list = degree_dist(H)
        * node degree dist = centrality_stats(degree_dist(H))
        * node degree hist = frequency_distribution(degree_dist(H))
        * max node degree = max(degree_dist(H))
        * edge size list = edge_size_dist(H)
        * edge size dist = centrality_stats(edge_size_dist(H))
        * edge size hist = frequency_distribution(edge_size_dist(H))
        * max edge size = max(edge_size_dist(H))
        * comp nodes list = s_comp_dist(H, s=1, edges=False)
        * comp nodes dist = centrality_stats(s_comp_dist(H, s=1, edges=False))
        * comp nodes hist = frequency_distribution(s_comp_dist(H, s=1, edges=False))
        * comp edges list = s_comp_dist(H, s=1, edges=True)
        * comp edges dist = centrality_stats(s_comp_dist(H, s=1, edges=True))
        * comp edges hist = frequency_distribution(s_comp_dist(H, s=1, edges=True))
        * num comps = len(s_comp_dist(H))
        
    Parameters
    ----------
    H : Hypergraph
    
    Returns
    -------
     dist_stats : dict
        Dictionary which keeps track of each of the above items (e.g., basic['nrows'] = the number of nodes in H)
    """

    basic = dict()
    
    # Number of rows (nodes), columns (edges), and aspect ratio
    basic['nrows'] = len(H.nodes)
    basic['ncols'] = len(H.edges)
    basic['aspect ratio'] = basic['nrows']/basic['ncols']
    
    # Number of cells and density
    M = H.incidence_matrix(index=False)
    basic['ncells'] = M.nnz
    basic['density'] = basic['ncells']/(basic['nrows']*basic['ncols'])
    
    # Node degree distribution
    basic['node degree list'] = sorted(degree_dist(H), reverse=True)
    basic['node degree dist'] = centrality_stats(basic['node degree list'])
    basic['node degree hist'] = frequency_distribution(basic['node degree list'])
    basic['max node degree'] = max(basic['node degree list'])
    
    # Edge size distribution
    basic['edge size list'] = sorted(edge_size_dist(H),reverse=True)
    basic['edge size dist'] = centrality_stats(basic['edge size list'])
    basic['edge size hist'] = frequency_distribution(basic['edge size list'])
    basic['max edge size'] = max(basic['edge size list'])
    
    # Component size distribution (nodes)
    basic['comp nodes list'] = sorted(s_comp_dist(H, edges=False),reverse=True)
    basic['comp nodes hist'] = frequency_distribution(basic['comp nodes list'])
    basic['comp nodes dist'] = centrality_stats(basic['comp nodes list'])
    
    # Component size distribution (edges)
    basic['comp edges list'] = sorted(s_comp_dist(H, edges=True), reverse=True)
    basic['comp edges hist'] = frequency_distribution(basic['comp edges list'])
    basic['comp edges dist'] = centrality_stats(basic['comp edges list'])
    
    # Number of components
    basic['num comps'] = len(basic['comp nodes list'])
    
    # # Diameters
    # basic['s edge diam list'] = s_edge_diameter_dist(H)
    # basic['s node diam list'] = s_node_diameter_dist(H)
    
    return basic
    


	