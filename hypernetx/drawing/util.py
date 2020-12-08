# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from itertools import combinations

import numpy as np

import networkx as nx

def inflate(items, v):
    if type(v) in {str, tuple, int, float}:
        return [v]*len(items)
    elif callable(v):
        return [v(i) for i in items]
    elif type(v) not in {list, np.ndarray} and hasattr(v, '__getitem__'):
        return [v[i] for i in items]
    return v

def inflate_kwargs(items, kwargs):
    '''
    Helper function to expand keyword arguments.

    Parameters
    ----------
    n: int
        length of resulting list if argument is expanded
    kwargs: dict
        keyword arguments to be expanded

    Returns
    -------
    dict
        dictionary with same keys as kwargs and whose values are lists of length n
    '''

    return {
        k: inflate(items, v)
        for k, v in kwargs.items()
    }

def transpose_inflated_kwargs(inflated):
    return [
        dict(zip(inflated, v))
        for v in zip(*inflated.values())
    ]

def get_frozenset_label(S, count=False, override={}):
    '''
    Helper function for rendering the labels of possibly collapsed nodes and edges

    Parameters
    ----------
    S: iterable
        list of entities to be labeled
    count: bool
        True if labels should be counts of entities instead of list

    Returns
    -------
    dict
        mapping of entity to its string representation
    '''
    def helper(v):
        if type(v) == frozenset:
            if count and len(v) > 1:
                return f'x {len(v)}'
            elif count:
                return ''
            else:
                return ', '.join([str(override.get(s, s)) for s in v])
        return str(v)

    return {v: override.get(v, helper(v)) for v in S}

def get_line_graph(H, collapse=True):
    '''
    Computes the line graph, a directed graph, where a directed edge (u, v)
    exists if the edge u is a subset of the edge v in the hypergraph.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    collapse: bool
        True if edges should be added if hyper edges are identical

    Returns
    -------
    networkx.DiGraph
        A directed graph
    '''
    D = nx.DiGraph()

    V = {edge: set(nodes)
         for edge, nodes in H.edges.elements.items()}
    
    D.add_nodes_from(V)

    for u, v in combinations(V, 2):
        if V[u] != V[v] or not collapse:
            if V[u].issubset(V[v]):
                D.add_edge(u, v)
            elif V[v].issubset(V[u]):
                D.add_edge(v, u)

    return D

def get_set_layering(H, collapse=True):
    '''
    Computes a layering of the edges in the hyper graph.

    In this layering, each edge is assigned a level. An edge u will be above
    (e.g., have a smaller level value) another edge v if v is a subset of u.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    collapse: bool
        True if edges should be added if hyper edges are identical

    Returns
    -------
    dict
        a mapping of vertices in H to integer levels
    '''

    D = get_line_graph(H, collapse=collapse)

    levels = {}

    for v in nx.topological_sort(D):
        parent_levels = [levels[u] for u, _ in D.in_edges(v)]
        levels[v] = max(parent_levels) + 1 if len(parent_levels) else 0

    return levels
