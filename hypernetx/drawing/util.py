# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from itertools import combinations

import numpy as np

import networkx as nx


def inflate(items, v):
    if type(v) in {str, tuple, int, float}:
        return [v] * len(items)
    elif callable(v):
        return [v(i) for i in items]
    elif type(v) not in {list, np.ndarray} and hasattr(v, "__getitem__"):
        return [v[i] for i in items]
    return v


def inflate_kwargs(items, kwargs):
    """
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
    """

    return {k: inflate(items, v) for k, v in kwargs.items()}


def transpose_inflated_kwargs(inflated):
    return [dict(zip(inflated, v)) for v in zip(*inflated.values())]


def get_collapsed_size(v):
    try:
        if type(v) == str and ":" in v:
            return int(v.split(":")[-1])
    except:
        pass

    return 1


def get_frozenset_label(S, count=False, override=None):
    """
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
    """

    if override is None:
        override = {}

    def helper(v):
        if type(v) == str:
            n = get_collapsed_size(v)
            if count and n > 1:
                return f"x {n}"
            elif count:
                return ""
        return str(v)

    return {v: override.get(v, helper(v)) for v in S}


def create_labels(
    equivalence_classes,
    with_counts=True,
    include_singletons=False,
    with_labels=True,
    as_set=False,
):
    """
    Convenience function to format labels for collapsed hyper graphs.

    Use hypernetx.Hypergraph.collapse_nodes(return_equivalence_classes=True),
    for example, to generate the collapsed hypergraph and equivalence classes
    to input to this function.

    Parameters
    ----------
    equivalence_classes: dict
        equivalence classes mapping an entity to a list of entities to be labeled
    with_counts: bool
        show the number of items in the equivalence class
    include_singletons: bool
        show the number even if the number of items is 1
    with_labels: bool
        show the representative of the equivalence class (the key in the dictionary)
    as_set: bool
        show the label as a set of all the members
    """

    def get_label(k, v):
        if as_set:
            return f'{{{", ".join(v)}}}'

        s = []

        if with_labels:
            s.append(str(k))

        if with_counts:
            n = len(v)
            if n > 1 or include_singletons:
                s.append(f"x{n}")

        return " ".join(s)

    return {k: get_label(k, v) for k, v in equivalence_classes.items()}


def get_line_graph(H, collapse=True):
    """
    Computes the line graph, a directed graph, where a directed edge (u, v)
    exists if the edge u is a subset of the edge v in the hypergraph.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    collapse: bool
        True if edges should be added if hyper edges are identical

    Returns
    -------
    networkx.DiGraph
        A directed graph
    """
    D = nx.DiGraph()

    V = {edge: set(nodes) for edge, nodes in H.edges.elements.items()}

    D.add_nodes_from(V)

    for u, v in combinations(V, 2):
        if V[u] != V[v] or not collapse:
            if V[u].issubset(V[v]):
                D.add_edge(u, v)
            elif V[v].issubset(V[u]):
                D.add_edge(v, u)

    return D


def get_set_layering(H, collapse=True):
    """
    Computes a layering of the edges in the hyper graph.

    In this layering, each edge is assigned a level. An edge u will be above
    (e.g., have a smaller level value) another edge v if v is a subset of u.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    collapse: bool
        True if edges should be added if hyper edges are identical

    Returns
    -------
    dict
        a mapping of vertices in H to integer levels
    """

    D = get_line_graph(H, collapse=collapse)

    levels = {}

    for v in nx.topological_sort(D):
        parent_levels = [levels[u] for u, _ in D.in_edges(v)]
        levels[v] = max(parent_levels) + 1 if len(parent_levels) else 0

    return levels


def layout_with_radius(B, node_and_edge_radius=1, **kwargs):
    """
    Convenience function allowing the user to specify ideal radii for nodes and edges in the drawing

    The Kamada-Kawai (networkx.kamada_kawai_layout) algorithm is used. The algorithm is passed
    a all pairs shorteset path matrix that is calculated from the bipartite graph input. The
    shortest path is determined


    Parameters
    ----------
    B: nx.Graph
        a bipartite graph representing the hypergraph
    node_and_edge_radius: float, int, dict, list, or function
        encoding of the radii of nodes in B, which are hyper-edges or hyper-nodes (0 by default)
    kwargs: dict
        Keyword arguments are passed through to the networkx.kamada_kawai_layout function

    Returns
    -------
    dict
        mapping of node and edge positions to R^2
    """

    # get radii encodings and convert to dictionary
    radius_dict = dict(zip(B, inflate(B, node_and_edge_radius)))

    # edges weights are the sum of the radii of the edge endpoints
    for u, v, d in B.edges(data=True):
        d["weight"] = radius_dict.get(u, 0) + radius_dict.get(v, 0)

    # compute all pairs shortest path (APSP)
    dist = dict(nx.all_pairs_dijkstra_path_length(B))

    # compute and return layout using above APSP; pass through arguments
    return nx.kamada_kawai_layout(B, dist=dist, **kwargs)
