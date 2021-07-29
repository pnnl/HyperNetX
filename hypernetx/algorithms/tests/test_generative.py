import numpy as np
import pytest
import hypernetx.algorithms.generative_models as gm
import hypernetx as hnx
import random
from scipy.sparse import coo_matrix

# Test the contagion functions
def test_chung_lu():
    random.seed(42)
    n = 1000
    k1 = {i : random.randint(1, 100) for i in range(n)}
    k2 = {i : sorted(k1.values())[i] for i in range(n)}
    H = gm.chung_lu_hypergraph(k1, k2)
    k1_true = {node:H.degree(node) for node in H.nodes}
    # makes sure that the sum of the degrees is within 1% of the desired sum of degrees
    assert abs((sum(k1_true.values()) - sum(k1.values()))/sum(k1.values())) <= 0.01
    # makes sure that the sum of the edge sizes is within 1% of the desired sum of edge sizes
    assert abs((sum(H.edge_size_dist()) - sum(k2.values()))/sum(k2.values())) <= 0.01
    # check that every degree is close (Within 15% on average)
    assert sum([abs(k1_true[node] - k1[node]) for node in H.nodes])/sum(k1.values()) <= 0.15


def test_dcsbm():
    random.seed(42)
    n = 1000
    k1 = {i : random.randint(1, 100) for i in range(n)}
    k2 = {i : sorted(k1.values())[i] for i in range(n)}
    g1 = {i : random.choice([0, 1]) for i in range(n)}
    g2 = {i : random.choice([0, 1]) for i in range(n)}
    omega = n*np.mean(list(k1.values()))*np.array([[0.45, 0.05], [0.05, 0.45]])
    H = gm.dcsbm_hypergraph(k1, k2, g1, g2, omega)
    k1_true = {node:H.degree(node) for node in H.nodes}
    # makes sure that the sum of the degrees is within 1% of the desired sum of degrees
    assert abs((sum(k1_true.values()) - sum(k1.values()))/sum(k1.values())) <= 0.01
    # makes sure that the sum of the edge sizes is within 1% of the desired sum of edge sizes
    assert abs((sum(H.edge_size_dist()) - sum(k2.values()))/sum(k2.values())) <= 0.01
    # check that every degree is close (Within 15% on average)
    assert sum([abs(k1_true[node] - k1[node]) for node in H.nodes])/sum(k1.values()) <= 0.15
    # I = coo_matrix(H.incidence_matrix())

    # for I