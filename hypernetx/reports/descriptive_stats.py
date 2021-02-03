"""
This module contains methods which compute various distributions for hypergraphs:
    * Edge size distribution
    * Node degree distribution
    * Component size distribution
    * Toplex size distribution
    * Diameter

Also computes general hypergraph information: number of nodes, edges, cells, aspect ratio, incidence matrix density
"""
from collections import Counter
import numpy as np
import networkx as nx
from hypernetx import *
from hypernetx.utils.decorators import not_implemented_for

__all__ = [
    'centrality_stats',
    'edge_size_dist',
    'degree_dist',
    'comp_dist',
    's_comp_dist',
    'toplex_dist',
    's_node_diameter_dist',
    's_edge_diameter_dist',
    'info',
    'info_dict',
    'dist_stats',
]


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
    if aggregated:
        return Counter(H.edge_size_dist())
    else:
        return H.edge_size_dist()


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
    if H.nwhy:
        distr = H.g.node_size_dist()
    else:
        distr = [H.degree(n) for n in H.nodes]
    if aggregated:
        return Counter(distr)
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
        return Counter(distr)
    else:
        return distr


def s_comp_dist(H, s=1, aggregated=False, edges=True, return_singletons=True):
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
    return_singletons : bool, optional, default=True

    Returns
    -------
     s_comp_dist : list or dictionary
        List of component sizes or dictionary of component size distribution in H

    See Also
    --------
    comp_dist

    '''
    distr = list()
    comps = H.s_connected_components(s=s, edges=edges, return_singletons=return_singletons)

    distr = [len(c) for c in comps]

    if aggregated:
        return Counter(distr)
    else:
        return distr


@not_implemented_for('static')
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
    distr = [H.size(e) for e in H.toplexes().edges]
    if aggregated:
        return Counter(distr)
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
    i = 1
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
    i = 1
    diams = []
    while H.is_connected(s=i, edges=True):
        diams.append(H.edge_diameter(s=i))
        i += 1
    return diams


def info(H, node=None, edge=None):
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
    if not H.edges.elements:
        return f"Hypergraph {H.name} is empty."
    report = info_dict(H, node=node, edge=edge)
    info = ''
    if node:
        info += f"Node '{node}' has the following properties:\n"
        info += f"Degree: {report['degree']}\n"
        info += f"Contained in: {report['membs']}\n"
        info += f"Neighbors: {report['neighbors']}"
    elif edge:
        info += f"Edge '{edge}' has the following properties:\n"
        info += f"Size: {report['size']}\n"
        info += f"Elements: {report['elements']}"
    else:
        info += f"Number of Rows: {report['nrows']}\n"
        info += f"Number of Columns: {report['ncols']}\n"
        info += f"Aspect Ratio: {report['aspect ratio']}\n"
        info += f"Number of non-empty Cells: {report['ncells']}\n"
        info += f"Density: {report['density']}"
    return info


def info_dict(H, node=None, edge=None):
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
    if len(H.edges.elements) == 0:
        return {}

    if node:
        report['membs'] = list(H.dual().edges[node])
        report['degree'] = len(report['membs'])
        report['neighbors'] = H.neighbors(node)
        return report
    if edge:
        report['size'] = H.size(edge)
        report['elements'] = list(H.edges[edge])
        return report
    else:
        lnodes, ledges = H.shape
        M = H.incidence_matrix(index=False)
        ncells = M.nnz

        report['nrows'] = lnodes
        report['ncols'] = ledges
        report['aspect ratio'] = lnodes / ledges
        report['ncells'] = ncells
        report['density'] = ncells / (lnodes * ledges)
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
        * node degree hist = Counter(degree_dist(H))
        * max node degree = max(degree_dist(H))
        * edge size list = edge_size_dist(H)
        * edge size dist = centrality_stats(edge_size_dist(H))
        * edge size hist = Counter(edge_size_dist(H))
        * max edge size = max(edge_size_dist(H))
        * comp nodes list = s_comp_dist(H, s=1, edges=False)
        * comp nodes dist = centrality_stats(s_comp_dist(H, s=1, edges=False))
        * comp nodes hist = Counter(s_comp_dist(H, s=1, edges=False))
        * comp edges list = s_comp_dist(H, s=1, edges=True)
        * comp edges dist = centrality_stats(s_comp_dist(H, s=1, edges=True))
        * comp edges hist = Counter(s_comp_dist(H, s=1, edges=True))
        * num comps = len(s_comp_dist(H))

    Parameters
    ----------
    H : Hypergraph

    Returns
    -------
     dist_stats : dict
        Dictionary which keeps track of each of the above items (e.g., basic['nrows'] = the number of nodes in H)
    """
    if H.isstatic and 'dist_stats' in H.state_dict:
        return H.state_dict['dist_stats']
    else:
        cstats = ['min', 'max', 'mean', 'median', 'std']
        basic = dict()

        # Number of rows (nodes), columns (edges), and aspect ratio
        basic['nrows'] = len(H.nodes)
        basic['ncols'] = len(H.edges)
        basic['aspect ratio'] = basic['nrows'] / basic['ncols']

        # Number of cells and density
        M = H.incidence_matrix(index=False)
        basic['ncells'] = M.nnz
        basic['density'] = basic['ncells'] / (basic['nrows'] * basic['ncols'])

        # Node degree distribution
        basic['node degree list'] = sorted(degree_dist(H), reverse=True)
        basic['node degree centrality stats'] = dict(zip(cstats, centrality_stats(basic['node degree list'])))
        basic['node degree hist'] = Counter(basic['node degree list'])
        basic['max node degree'] = max(basic['node degree list'])

        # Edge size distribution
        basic['edge size list'] = sorted(H.edge_size_dist(), reverse=True)
        basic['edge size centrality stats'] = dict(zip(cstats, centrality_stats(basic['edge size list'])))
        basic['edge size hist'] = Counter(basic['edge size list'])
        basic['max edge size'] = max(basic['edge size hist'])

        # Component size distribution (nodes)
        basic['comp nodes list'] = sorted(s_comp_dist(H, edges=False), reverse=True)
        basic['comp nodes hist'] = Counter(basic['comp nodes list'])
        basic['comp nodes centrality stats'] = dict(zip(cstats, centrality_stats(basic['comp nodes list'])))

        # Component size distribution (edges)
        basic['comp edges list'] = sorted(s_comp_dist(H, edges=True), reverse=True)
        basic['comp edges hist'] = Counter(basic['comp edges list'])
        basic['comp edges centrality stats'] = dict(zip(cstats, centrality_stats(basic['comp edges list'])))

        # Number of components
        basic['num comps'] = len(basic['comp nodes list'])

        # # Diameters
        # basic['s edge diam list'] = s_edge_diameter_dist(H)
        # basic['s node diam list'] = s_node_diameter_dist(H)
        if H.isstatic:
            H.set_state(dist_stats=basic)
        return basic
