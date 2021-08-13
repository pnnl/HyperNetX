import numpy as np
import pytest
import hypernetx.algorithms.generative_models as gm
import hypernetx as hnx
import random
from scipy.sparse import coo_matrix

# Test the generative model functions
def test_erdos_renyi():
    random.seed(42)
    n = 1000
    m = 1000
    p = 0.01
    H = gm.erdos_renyi_hypergraph(n, m, p)
    # makes sure that the number of bipartite edges is within 1% of desired
    assert abs(n * m * p - H.incidence_matrix().count_nonzero()) / (n * m * p) <= 0.01
    assert H.shape == (n, m)


def test_chung_lu():
    random.seed(42)
    n = 1000
    m = 1000
    k1 = {i: random.randint(10, 30) for i in range(n)}
    k2 = {i: sorted(k1.values())[i] for i in range(m)}
    H = gm.chung_lu_hypergraph(k1, k2)
    k1_true = {node: H.degree(node) for node in H.nodes}
    # makes sure that the sum of the degrees is within 1% of the desired sum of degrees
    assert abs((sum(k1_true.values()) - sum(k1.values())) / sum(k1.values())) <= 0.01
    # makes sure that the sum of the edge sizes is within 1% of the desired sum of edge sizes
    assert abs((sum(H.edge_size_dist()) - sum(k2.values())) / sum(k2.values())) <= 0.01
    # check that every degree is close (Within 15% on average)
    assert (
        sum([abs(k1_true[node] - k1[node]) for node in H.nodes]) / sum(k1.values())
        <= 0.18
    )


def test_dcsbm():
    random.seed(42)
    n = 1000
    m = 1000
    k1 = {i: random.randint(10, 30) for i in range(n)}
    k2 = {i: sorted(k1.values())[i] for i in range(m)}
    g1 = {i: random.choice([0, 1]) for i in range(n)}
    g2 = {i: random.choice([0, 1]) for i in range(m)}
    omega = n * np.mean(list(k1.values())) * np.array([[0.45, 0.05], [0.05, 0.45]])
    H = gm.dcsbm_hypergraph(k1, k2, g1, g2, omega)
    k1_true = {node: H.degree(node) for node in H.nodes}
    # makes sure that the sum of the degrees is within 1% of the desired sum of degrees
    assert abs((sum(k1_true.values()) - sum(k1.values())) / sum(k1.values())) <= 0.01
    # makes sure that the sum of the edge sizes is within 1% of the desired sum of edge sizes
    assert abs((sum(H.edge_size_dist()) - sum(k2.values())) / sum(k2.values())) <= 0.01
    # check that every degree is close (Within 15% on average)
    assert (
        sum([abs(k1_true[node] - k1[node]) for node in H.nodes]) / sum(k1.values())
        <= 0.18
    )
    I, rows, cols = H.incidence_matrix(index=True)
    I = coo_matrix(I)

    in_community_edges = 0
    between_community_edges = 0
    for i in range(len(I.row)):
        if g1[rows[I.row[i]]] == g2[cols[I.col[i]]]:
            in_community_edges += 1
        else:
            between_community_edges += 1

    # The target ratio of in-community edges to between-community edges is 10. We check that it's at least 5.
    assert in_community_edges > 5 * between_community_edges

    omega = n * np.mean(list(k1.values())) * np.array([[0.5, 0], [0, 0.5]])
    H = gm.dcsbm_hypergraph(k1, k2, g1, g2, omega)

    k1_true = {node: H.degree(node) for node in H.nodes}
    # makes sure that the sum of the degrees is within 1% of the desired sum of degrees
    assert abs((sum(k1_true.values()) - sum(k1.values())) / sum(k1.values())) <= 0.01
    # makes sure that the sum of the edge sizes is within 1% of the desired sum of edge sizes
    assert abs((sum(H.edge_size_dist()) - sum(k2.values())) / sum(k2.values())) <= 0.01
    # check that every degree is close (Within 15% on average)
    assert (
        sum([abs(k1_true[node] - k1[node]) for node in H.nodes]) / sum(k1.values())
        <= 0.18
    )
    I, rows, cols = H.incidence_matrix(index=True)
    I = coo_matrix(I)

    in_community_edges = 0
    between_community_edges = 0
    for i in range(len(I.row)):
        if g1[rows[I.row[i]]] == g2[cols[I.col[i]]]:
            in_community_edges += 1
        else:
            between_community_edges += 1
    assert between_community_edges == 0
