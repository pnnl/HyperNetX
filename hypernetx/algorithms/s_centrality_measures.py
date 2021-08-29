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
from functools import partial

try:
    import nwhy

    nwhy_available = True
except:
    nwhy_available = False

sys.setrecursionlimit(10000)

__all__ = [
    "s_betweenness_centrality",
    "s_harmonic_closeness_centrality",
    "s_harmonic_centrality",
    "s_closeness_centrality",
    "s_eccentricity",
]


def _s_centrality(
    func, H, s=1, edges=True, f=None, return_singletons=True, use_nwhy=True, **kwargs
):
    """
    Wrapper for computing s-centrality either in NetworkX or in NWHy

    Parameters
    ----------
    func : function
        Function or partial function from NetworkX or NWHy
    H : hnx.Hypergraph

    s : int, optional
        s-width for computation
    edges : bool, optional
        If True, an edge linegraph will be used, otherwise a node linegraph will be used
    f : str, optional
        Identifier of node or edge of interest for computing centrality
    return_singletons : bool, optional
        If True will return 0 value for each singleton in the s-linegraph
    use_nwhy : bool, optional
        If True will attempt to use nwhy centrality methods if availaable
    **kwargs
        Centrality metric specific keyword arguments to be passed to func

    Returns
    -------
    dict
        dictionary of centrality scores keyed by names
    """
    comps = H.s_component_subgraphs(
        s=s, edges=edges, return_singletons=return_singletons
    )
    if f is not None:
        for cps in comps:
            if (edges and f in cps.edges) or (not edges and f in cps.nodes):
                comps = [cps]
                break
        else:
            return {f: 0}

    stats = dict()
    if H.isstatic:
        for h in comps:
            if edges:
                vertices = h.edges
            else:
                vertices = h.nodes

            if h.shape[edges * 1] == 1:
                stats.update({v: 0 for v in vertices})
            elif use_nwhy and nwhy_available and h.nwhy:
                g = h.get_linegraph(s=s, edges=edges, use_nwhy=True)
                stats.update(dict(zip(vertices, func(g, **kwargs))))
            else:
                g = h.get_linegraph(s=s, edges=edges, use_nwhy=False)
                stats.update(
                    {
                        h.get_name(k, edges=edges): v
                        for k, v in func(g, **kwargs).items()
                    }
                )
            if f:
                return {f: stats[f]}
    else:
        for h in comps:
            if edges:
                A, Adict = h.edge_adjacency_matrix(s=s, index=True)
            else:
                A, Adict = h.adjacency_matrix(s=s, index=True)
            g = nx.from_scipy_sparse_matrix(A)
            stats.update({Adict[k]: v for k, v in func(g, **kwargs).items()})
            if f:
                return {f: stats[f]}

    return stats


def s_betweenness_centrality(
    H, s=1, edges=True, normalized=True, return_singletons=True, use_nwhy=True
):
    r"""
    A centrality measure for an s-edge(node) subgraph of H based on shortest paths.
    Equals the betweenness centrality of vertices in the edge(node) s-linegraph.

    In a graph (2-uniform hypergraph) the betweenness centrality of a vertex $v$
    is the ratio of the number of non-trivial shortest paths between any pair of
    vertices in the graph that pass through $v$ divided by the total number of
    non-trivial shortest paths in the graph.

    The centrality of edge to all shortest s-edge paths
    $V$ = the set of vertices in the linegraph.
    $\sigma(s,t)$ = the number of shortest paths between vertices $s$ and $t$.
    $\sigma(s,t|v)$ = the number of those paths that pass through vertex $v$.

    .. math::

        c_B(v) = \sum_{s \neq t \in V} \frac{\sigma(s, t|v)}{\sigma(s,t)}

    Parameters
    ----------
    H : hnx.Hypergraph
    s : int
        s connectedness requirement
    edges : bool, optional
        determines if edge or node linegraph
    normalized
        bool, default=False,
        If true the betweenness values are normalized by `2/((n-1)(n-2))`,
        where n is the number of edges in H
    return_singletons : bool, optional
        if False will ignore singleton components of linegraph

    Returns
    -------
    dict
        A dictionary of s-betweenness centrality value of the edges.

    """
    if use_nwhy and nwhy_available and H.nwhy:
        func = partial(nwhy.Slinegraph.s_betweenness_centrality, normalized=False)
    else:
        use_nwhy = False
        func = partial(nx.betweenness_centrality, normalized=False)
    result = _s_centrality(
        func,
        H,
        s=s,
        edges=edges,
        return_singletons=return_singletons,
        use_nwhy=use_nwhy,
    )

    if normalized and H.shape[edges * 1] > 2:
        n = H.shape[edges * 1]
        return {k: v * 2 / ((n - 1) * (n - 2)) for k, v in result.items()}
    else:
        return result


def s_closeness_centrality(
    H, s=1, edges=True, return_singletons=True, source=None, use_nwhy=True
):
    r"""
    In a connected component the reciprocal of the sum of the distance between an
    edge(node) and all other edges(nodes) in the component times the number of edges(nodes)
    in the component minus 1.

    $V$ = the set of vertices in the linegraph.
    $n = |V|$
    $d$ = shortest path distance

    .. math::

        C(u) = \frac{n - 1}{\sum_{v \neq u \in V} d(v, u)}


    Parameters
    ----------
    H : hnx.Hypergraph

    s : int, optional

    edges : bool, optional
        Indicates if method should compute edge linegraph (default) or node linegraph.
    return_singletons : bool, optional
        Indicates if method should return values for singleton components.
    source : str, optional
        Identifier of node or edge of interest for computing centrality
    use_nwhy : bool, optional
        If true will use the :ref:`NWHy library <nwhy>` if available.

    Returns
    -------
    dict or float
        returns the s-closeness centrality value of the edges(nodes).
        If source=None a dictionary of values for each s-edge in H is returned.
        If source then a single value is returned.
    """
    if use_nwhy and nwhy_available and H.nwhy:
        func = partial(nwhy.Slinegraph.s_closeness_centrality)
    else:
        use_nwhy = False
        func = partial(nx.closeness_centrality)
    return _s_centrality(
        func,
        H,
        s=s,
        edges=edges,
        return_singletons=return_singletons,
        f=source,
        use_nwhy=use_nwhy,
    )


def s_harmonic_closeness_centrality(H, s=1, edge=None, use_nwhy=True):
    msg = """
    s_harmonic_closeness_centrality is being replaced with s_harmonic_centrality 
    and will not be available in future releases. 
    """
    warnings.warn(msg)
    return s_harmonic_centrality(H, s=s, edges=True, normalized=True, source=edge)


def s_harmonic_centrality(
    H,
    s=1,
    edges=True,
    source=None,
    normalized=False,
    return_singletons=True,
    use_nwhy=True,
):
    r"""
    A centrality measure for an s-edge subgraph of H. A value equal to 1 means the s-edge
    intersects every other s-edge in H. All values range between 0 and 1.
    Edges of size less than s return 0. If H contains only one s-edge a 0 is returned.

    The denormalized reciprocal of the harmonic mean of all distances from $u$ to all other vertices.
    $V$ = the set of vertices in the linegraph.
    $d$ = shortest path distance

    .. math::

        C(u) = \sum_{v \neq u \in V} \frac{1}{d(v, u)}

    Normalized this becomes:
    $$C(u) = \sum_{v \neq u \in V} \frac{1}{d(v, u)}\cdot\frac{2}{(n-1)(n-2)}$$
    where $n$ is the number vertices.

    Parameters
    ----------
    H : hnx.Hypergraph

    s : int, optional

    edges : bool, optional
        Indicates if method should compute edge linegraph (default) or node linegraph.
    return_singletons : bool, optional
        Indicates if method should return values for singleton components.
    source : str, optional
        Identifier of node or edge of interest for computing centrality
    use_nwhy : bool, optional
        If true will use the :ref:`NWHy library <nwhy>` if available.

    Returns
    -------
    dict or float
        returns the s-harmonic closeness centrality value of the edges, a number between 0 and 1 inclusive.
        If source=None a dictionary of values for each s-edge in H is returned.
        If source then a single value is returned.

    """

    if use_nwhy and nwhy_available and H.nwhy:
        func = partial(nwhy.Slinegraph.s_harmonic_closeness_centrality)
    else:
        use_nwhy = False
        func = partial(nx.harmonic_centrality)
    result = _s_centrality(
        func,
        H,
        s=s,
        edges=edges,
        return_singletons=return_singletons,
        f=source,
        use_nwhy=use_nwhy,
    )

    if normalized and H.shape[edges * 1] > 2:
        n = H.shape[edges * 1]
        return {k: v * 2 / ((n - 1) * (n - 2)) for k, v in result.items()}
    else:
        return result


def s_eccentricity(
    H, s=1, edges=True, source=None, return_singletons=True, use_nwhy=True
):
    r"""
    The length of the longest shortest path from a vertex $u$ to every other vertex in the linegraph.
    $V$ = set of vertices in the linegraph
    $d$ = shortest path distance

    .. math::

        \text{s-ecc}(u) = \text{max}\{d(u,v): v \in V\}

    Parameters
    ----------
    H : hnx.Hypergraph

    s : int, optional

    edges : bool, optional
        Indicates if method should compute edge linegraph (default) or node linegraph.
    return_singletons : bool, optional
        Indicates if method should return values for singleton components.
    source : str, optional
        Identifier of node or edge of interest for computing centrality
    use_nwhy : bool, optional
        If true will use the :ref:`NWHy library <nwhy>` if available.

    Returns
    -------
    dict or float
        returns the s-eccentricity value of the edges(nodes).
        If source=None a dictionary of values for each s-edge in H is returned.
        If source then a single value is returned.

    """
    if use_nwhy and nwhy_available and H.nwhy:
        func = nwhy.Slinegraph.s_eccentricity
    else:
        use_nwhy = False
        func = nx.eccentricity

    if source is not None:
        return _s_centrality(
            func,
            H,
            s=s,
            edges=edges,
            f=source,
            return_singletons=return_singletons,
            use_nwhy=use_nwhy,
        )
    else:
        return _s_centrality(
            func,
            H,
            s=s,
            edges=edges,
            return_singletons=return_singletons,
            use_nwhy=use_nwhy,
        )
