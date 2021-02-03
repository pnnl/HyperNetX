# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

"""

S-Centrality Measures
=====================
We generalize graph metrics to s-metrics for a hypergraph by using its s-connected
components. This is accomplished by computing the s edge-adjacency matrix and
constructing the corresponding graph of the matrix. We then use existing graph metrics
on this representation of the hypergraph. In essence we construct an *s*-line graph
corresponding to the hypergraph on which to apply our methods.

S-Metrics for hypergraphs are discussed in depth in:        
*Aksoy, S.G., Joslyn, C., Ortiz Marrero, C. et al. Hypernetwork science via high-order hypergraph walks.
EPJ Data Sci. 9, 16 (2020). https://doi.org/10.1140/epjds/s13688-020-00231-0*

"""

import numpy as np
from collections import defaultdict
import networkx as nx
import warnings
import sys
sys.setrecursionlimit(10000)

__all__ = [
    's_betweenness_centrality',
    's_harmonic_closeness_centrality',
    's_eccentricity',
]


def s_betweenness_centrality(H, s=1, normalized=True):
    '''
    A centrality measure for an s-edge subgraph of H based on shortest paths.
    The betweenness centrality of an s-edge e is the sum of the fraction of all
    shortest s-paths between s-edges that pass through e.

    Parameters:
    -----------
    H : Hypergraph
    s : int
        minimum size of edges to be considered
    normalized : bool, default=False,
        If true the betweenness values are normalized by `2/((n-1)(n-2))`,
        where n is the number of edges in H

    Returns:
    --------
     : dict
        A dictionary of s-betweenness centrality value of the edges.

    '''

    # Confirm there is at least 1 s edge for which we can compute the centrality
    # Find all s-edges
    #M,rdict,_ = H.incidence_matrix(index=True)
    #A = M.transpose().dot(M)

    A, coldict = H.edge_adjacency_matrix(s=s, index=True)
    A = (A >= s) * 1
    g = nx.from_scipy_sparse_matrix(A)
    return {coldict[k]: v for k, v in nx.betweenness_centrality(g, normalized=normalized).items()}


def s_harmonic_closeness_centrality(H, edge=None, s=1):
    '''
    A centrality measure for an s-edge subgraph of H. A value equal to 1 means the s-edge
    intersects every other s-edge in H. All values range between 0 and 1.
    Edges of size less than s return 0. If H contains only one s-edge a 0 is returned.

    Parameters:
    -----------
    H : Hypergraph
    edge : str or Entity, optional
        an edge or uid of an edge in H
        If None then a dictionary of values for all s-edges is returned.
    s : int
        minimum size of edges to be considered

    Returns:
    --------
     : dict or float
        returns the s-harmonic closeness centrality value of the edges, a number between 0 and 1 inclusive.
        If edge=None a dictionary of values for each s-edge in H is returned.
        If edge then a single value is returned.

    '''

    # Confirm there is at least 1 s edge for which we can compute the centrality
    # Find all s-edges

    if edge and len(H.edges[edge]) < s:
        return 0

    Es = [e for e in H.edges if len(H.edges[e]) >= s]
    if edge:
        edges = [H.edges[edge].uid]
    else:
        edges = Es

    A, coldict = H.edge_adjacency_matrix(s=s, index=True)
    g = nx.from_scipy_sparse_matrix(A)
    ckey = {v: k for k, v in coldict.items()}

    def temp(e, f):
        try:
            return nx.shortest_path_length(g, ckey[e], ckey[f])
        except:
            return np.inf

    # confirm there are at least 2 s-edges
    # we follow the NX convention that the s-closeness centrality of a single edge Hypergraph is 0

    output = {}
    if not bool(Es) or len(Es) == 1:
        output = {e: 0 for e in edges}
    else:
        for e in edges:
            summands_recip = [temp(e, f) for f in Es if f != e]
            summands = [1 / x for x in summands_recip if not x == np.inf and x != 0]
            output[e] = 1 / (len(Es) - 1) * sum(summands)
    if len(edges) == 1:
        return output[edges[0]]
    else:
        return output


def s_eccentricity(H, f=None, s=1):
    '''
    Max s_distance from edge f to every other edge to which it is connected

    Parameters:
    -----------
    H : Hypergraph
    f : Entity or str
    s : int

    Returns:
    --------
    if f:
        eccentricity[f] : float
    else:
        eccentricity_dict : dict
     : dict or float
        returns the s-eccentricity value of the edges, a floating point number
        If edge=None a dictionary of values for each s-edge in H is returned.
        If edge then a single value is returned.

    '''
    if f:
        if isinstance(f, Entity):
            source = [f.uid]
        else:
            source = [f]
    else:
        source = H.edges

    eccentricity_dict = defaultdict()

    A, coldict = H.edge_adjacency_matrix(s=s, index=True)
    g = nx.from_scipy_sparse_matrix(A)
    ckey = {v: k for k, v in coldict.items()}
    for sedge in source:
        ecclist = []
        for e in H.edges:
            try:
                ecclist.append(nx.shortest_path_length(g, ckey[sedge], ckey[e]))
            except:
                pass
        eccentricity_dict[sedge] = np.max(ecclist)

    if f:
        return eccentricity_dict[f]
    else:
        return dict(eccentricity_dict)
