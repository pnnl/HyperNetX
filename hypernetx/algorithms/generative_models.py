import random
import math
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from hypernetx import Hypergraph


def erdos_renyi_hypergraph(n, m, p, node_labels=None, edge_labels=None):
    """
    A function to generate an Erdos-Renyi hypergraph as implemented by Mirah Shi and described for
    bipartite networks by Aksoy et al. in https://doi.org/10.1093/comnet/cnx001

    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges
    p: float
        The probability that a bipartite edge is created
    node_labels: list, default=None
        Vertex labels
    edge_labels: list, default=None
        Hyperedge labels

    Returns
    -------
    HyperNetX Hypergraph object


    Example::
    
    >>> import hypernetx.algorithms.generative_models as gm
    >>> n = 1000
    >>> m = n
    >>> p = 0.01
    >>> H = gm.erdos_renyi_hypergraph(n, m, p)

    """

    if node_labels is not None and edge_labels is not None:
        get_node_label = lambda index: node_labels[index]
        get_edge_label = lambda index: edge_labels[index]
    else:
        get_node_label = lambda index: index
        get_edge_label = lambda index: index

    bipartite_edges = []
    for u in range(n):
        v = 0
        while v < m:
            # identify next pair
            r = random.random()
            v = v + math.floor(math.log(r) / math.log(1 - p))
            if v < m:
                # add vertex hyperedge pair
                bipartite_edges.append((get_edge_label(v), get_node_label(u)))
                v = v + 1

    df = pd.DataFrame(bipartite_edges)
    return Hypergraph(df, static=True)


def chung_lu_hypergraph(k1, k2):
    """
    A function to generate an extension of Chung-Lu hypergraph as implemented by Mirah Shi and described for
    bipartite networks by Aksoy et al. in https://doi.org/10.1093/comnet/cnx001

    Parameters
    ----------
    k1 : dictionary
        This a dictionary where the keys are node ids and the values are node degrees.
    k2 : dictionary
        This a dictionary where the keys are edge ids and the values are edge degrees also known as edge sizes.
    Returns
    -------
    HyperNetX Hypergraph object

    Notes
    -----
    The sums of k1 and k2 should be roughly the same. If they are not the same, this function returns a warning but still runs.
    The output currently is a static Hypergraph object. Dynamic hypergraphs are not currently supported.

    Example::

    >>> import hypernetx.algorithms.generative_models as gm
    >>> import random
    >>> n = 100
    >>> k1 = {i : random.randint(1, 100) for i in range(n)}
    >>> k2 = {i : sorted(k1.values())[i] for i in range(n)}
    >>> H = gm.chung_lu_hypergraph(k1, k2)
    """

    # sort dictionary by degree in decreasing order
    Nlabels = [n for n, _ in sorted(k1.items(), key=lambda d: d[1], reverse=True)]
    Mlabels = [m for m, _ in sorted(k2.items(), key=lambda d: d[1], reverse=True)]

    m = len(k2)

    if sum(k1.values()) != sum(k2.values()):
        warnings.warn(
            "The sum of the degree sequence does not match the sum of the size sequence"
        )

    S = sum(k1.values())

    bipartite_edges = []
    for u in Nlabels:
        j = 0
        v = Mlabels[j]  # start from beginning every time
        p = min((k1[u] * k2[v]) / S, 1)

        while j < m:
            if p != 1:
                r = random.random()
                j = j + math.floor(math.log(r) / math.log(1 - p))
            if j < m:
                v = Mlabels[j]
                q = min((k1[u] * k2[v]) / S, 1)
                r = random.random()
                if r < q / p:
                    # no duplicates
                    bipartite_edges.append((v, u))

                p = q
                j = j + 1

    df = pd.DataFrame(bipartite_edges)
    return Hypergraph(df, static=True)


def dcsbm_hypergraph(k1, k2, g1, g2, omega):
    """
    A function to generate an extension of DCSBM hypergraph as implemented by Mirah Shi and described for
    bipartite networks by Larremore et al. in https://doi.org/10.1103/PhysRevE.90.012805

    Parameters
    ----------
    k1 : dictionary
        This a dictionary where the keys are node ids and the values are node degrees.
    k2 : dictionary
        This a dictionary where the keys are edge ids and the values are edge degrees also known as edge sizes.
    g1 : dictionary
        This a dictionary where the keys are node ids and the values are the group ids to which the node belongs.
        The keys must match the keys of k1.
    g2 : dictionary
        This a dictionary where the keys are edge ids and the values are the group ids to which the edge belongs.
        The keys must match the keys of k2.
    omega : 2D numpy array
        This is a matrix with entries which specify the number of edges between a given node community and edge community.
        The number of rows must match the number of node communities and the number of columns
        must match the number of edge communities.


    Returns
    -------
    HyperNetX Hypergraph object

    Notes
    -----
    The sums of k1 and k2 should be the same. If they are not the same, this function returns a warning but still runs.
    The sum of k1 (and k2) and omega should be the same. If they are not the same, this function returns a warning
    but still runs and the number of entries in the incidence matrix is determined by the omega matrix.

    The output currently is a static Hypergraph object. Dynamic hypergraphs are not currently supported.

    Example::

    >>> n = 100
    >>> k1 = {i : random.randint(1, 100) for i in range(n)}
    >>> k2 = {i : sorted(k1.values())[i] for i in range(n)}
    >>> g1 = {i : random.choice([0, 1]) for i in range(n)}
    >>> g2 = {i : random.choice([0, 1]) for i in range(n)}
    >>> omega = np.array([[100, 10], [10, 100]])
    >>> H = gm.dcsbm_hypergraph(k1, k2, g1, g2, omega)
    """

    # sort dictionary by degree in decreasing order
    Nlabels = [n for n, _ in sorted(k1.items(), key=lambda d: d[1], reverse=True)]
    Mlabels = [m for m, _ in sorted(k2.items(), key=lambda d: d[1], reverse=True)]

    # these checks verify that the sum of node and edge degrees and the sum of node degrees
    # and the sum of community connection matrix differ by less than a single edge.
    if abs(sum(k1.values()) - sum(k2.values())) > 1:
        warnings.warn(
            "The sum of the degree sequence does not match the sum of the size sequence"
        )

    if abs(sum(k1.values()) - np.sum(omega)) > 1:
        warnings.warn(
            "The sum of the degree sequence does not match the entries in the omega matrix"
        )

    # get indices for each community
    community1Indices = defaultdict(list)
    for label in Nlabels:
        group = g1[label]
        community1Indices[group].append(label)

    community2Indices = defaultdict(list)
    for label in Mlabels:
        group = g2[label]
        community2Indices[group].append(label)

    bipartite_edges = list()

    kappa1 = defaultdict(lambda: 0)
    kappa2 = defaultdict(lambda: 0)
    for id, g in g1.items():
        kappa1[g] += k1[id]
    for id, g in g2.items():
        kappa2[g] += k2[id]

    for group1 in community1Indices.keys():
        for group2 in community2Indices.keys():
            # for each constant probability patch
            try:
                groupConstant = omega[group1, group2] / (
                    kappa1[group1] * kappa2[group2]
                )
            except:
                groupConstant = 0

            for u in community1Indices[group1]:
                j = 0
                v = community2Indices[group2][j]  # start from beginning every time
                # max probability
                p = min(k1[u] * k2[v] * groupConstant, 1)
                while j < len(community2Indices[group2]):
                    if p != 1:
                        r = random.random()
                        try:
                            j = j + math.floor(math.log(r) / math.log(1 - p))
                        except:
                            j = np.inf
                    if j < len(community2Indices[group2]):
                        v = community2Indices[group2][j]
                        q = min((k1[u] * k2[v]) * groupConstant, 1)
                        r = random.random()
                        if r < q / p:
                            # no duplicates
                            bipartite_edges.append((v, u))

                            p = q
                            j = j + 1

    df = pd.DataFrame(bipartite_edges)
    return Hypergraph(df, static=True)
