import numpy as np
import scipy
import hypernetx as hnx
from hypernetx.algorithms.concepts import HypergraphLattice


# Helper function to create a simple hypergraph for testing
def test_lattice():
    # Example hypergraph lattice
    d = {
        "0": ("a", "b", "c"),
        "1": ("b", "c", "d"),
        "2": ("a", "c", "d"),
        "3": ("a", "b", "d"),
    }
    H = hnx.Hypergraph(d)
    return HypergraphLattice(H)


# Test: Initialize HypergraphLattice and check properties
def test_hypergraph_lattice_init():
    lat = test_lattice()
    assert len(lat._concepts) == 16
    assert lat[-1] in lat[0].upset()


def test_distance():
    lat = test_lattice()

    x = lat[2]
    y = lat[10]

    weights = {"a": 1, "b": 2, "c": 3, "d": 4}

    assert lat.distance(x, y, metric="shortest_path") == 3.0
    assert lat.distance(x, y, metric="upper_valuation") == 8.0
    assert lat.distance(x, y, metric="lower_valuation") == 4.0
    assert lat.distance(x, y, metric=weights) == 9.0


# Helper function for converting distance dictionaries into np.arrays
def dist_matrix(dist_dict):
    num_nodes = len(dist_dict)
    d = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d[i, j] = dist_dict[i][j]
    return d


def test_all_distances():
    lat = test_lattice()
    weights = {"a": 1, "b": 2, "c": 3, "d": 4}

    sp_mat = dist_matrix(lat.all_distances(metric="shortest_path"))
    uv_mat = dist_matrix(lat.all_distances(metric="upper_valuation"))
    lv_mat = dist_matrix(lat.all_distances(metric="lower_valuation"))
    weight_mat = dist_matrix(lat.all_distances(metric=weights))

    assert scipy.spatial.distance.is_valid_dm(sp_mat)
    assert scipy.spatial.distance.is_valid_dm(uv_mat)
    assert scipy.spatial.distance.is_valid_dm(lv_mat)
    assert scipy.spatial.distance.is_valid_dm(weight_mat)
