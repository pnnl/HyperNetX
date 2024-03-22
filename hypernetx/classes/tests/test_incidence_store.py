import numpy as np
import pandas as pd
from hypernetx.classes.incidence_store import IncidenceStore
import pytest


def test_iter():
    # Test iteration over incidence pairs
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)
    expected_pairs = [(1, 1), (2, 2), (2, 3), (3, 4)]
    actual_pairs = list(store)

    assert len(actual_pairs) == len(expected_pairs)
    for i in range(len(actual_pairs)):
        assert actual_pairs[i] == expected_pairs[i]


def test_len():
    # Test length of incidence store
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert len(store) == 4  # Should match the number of unique pairs


def test_contains():
    # Test if incidence pair exists
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)

    assert (1, 1) in store
    assert (2, 2) in store
    assert (2, 3) in store
    assert (3, 4) in store
    assert (3, 5) not in store  # Non-existent pair


def test_neighbors():
    # Test getting elements or memberships based on level and key
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)

    assert store.neighbors(0, 1) == [1]  # Nodes in edge 1
    assert store.neighbors(1, 2) == [2, 3]  # Edges containing node 2
    assert store.neighbors(0, 5) == []  # Non-existent edge

    assert store.neighbors(3, 1) == []


def test_edges():
    # Test getting all edges
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    expected_edges = np.array([1, 2, 3, 4])
    store = IncidenceStore(data)
    assert np.array_equal(store.edges, expected_edges)


def test_nodes():
    # Test getting all nodes
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    expected_nodes = np.array([1, 2, 3])
    store = IncidenceStore(data)
    assert np.array_equal(store.nodes, expected_nodes)


def test_dimensions():
    # Test getting number of nodes and edges
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert store.dimensions == (4, 3)  # (4 unique edges, 3 unique nodes)


def test_restrict_to():
    # Test restricting to a subset based on level and items
    data = pd.DataFrame({"nodes": [1, 2, 2, 3], "edges": [1, 2, 3, 4]})
    store = IncidenceStore(data)

    # Inplace restriction
    store.restrict_to(0, [1, 2], inplace=True)
    assert store._data.equals(pd.DataFrame({"nodes": [1, 2], "edges": [1, 2]}))
    store = IncidenceStore(data)  # Recreate initial store

    # Non-inplace restriction (returns new dataframe)
    restricted_df = store.restrict_to(0, [1, 2], inplace=False)
    # Should be a new dataframe
    # check that restricted_df and store._data are separate objects (i.e. object reference comparison)
    assert restricted_df is not (store._data)
    # check that restricted_df has expected values (i.e. value comparison)
    assert restricted_df.equals(pd.DataFrame({"nodes": [1, 2], "edges": [1, 2]}))

    # Invalid level
    with pytest.raises(ValueError):
        store.restrict_to(3, [1])  # Invalid level should raise error

    # Non-existent items
    store = IncidenceStore(data)
    restricted_df = store.restrict_to(0, [5], inplace=False)
    assert restricted_df.empty  # Empty dataframe as no pairs with item 5
