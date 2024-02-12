import pandas as pd
from incidence_store import IncidenceStore
import pytest

def test_iter():
    # Test iteration over incidence pairs
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)
    expected_pairs = [(1, 1), (2, 2), (2, 3), (3, 4)]
    actual_pairs = list(store)

    assert len(actual_pairs) == len(expected_pairs)
    for i in range(len(actual_pairs)):
        assert actual_pairs[i] == expected_pairs[i]

def test_len():
    # Test length of incidence store
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert len(store) == 4  # Should match the number of unique pairs

def test_contains():
    # Test if incidence pair exists
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)

    assert (1, 1) in store
    assert (2, 3) in store
    assert (3, 5) not in store  # Non-existent pair

def test_neighbors():
    # Test getting elements or memberships based on level and key
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)

    assert store.neighbors(0, 1) == [1]  # Nodes in edge 1
    assert store.neighbors(1, 2) == [2, 3]  # Edges containing node 2
    assert store.neighbors(0, 5) == []  # Non-existent edge

    with pytest.raises(ValueError):
        store.neighbors(3, 1)  # Invalid level

def test_edges():
    # Test getting all edges
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert store.edges() == [1, 2, 3, 4]

def test_nodes():
    # Test getting all nodes
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert store.nodes() == [1, 2, 3]

def test_dimensions():
    # Test getting number of nodes and edges
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)
    assert store.dimensions() == (3, 4)  # (3 unique nodes, 4 unique edges)

def test_restrict_to():
    # Test restricting to a subset based on level and items
    data = pd.DataFrame({'nodes': [1, 2, 2, 3], 'edges': [1, 2, 3, 4]})
    store = IncidenceStore(data)

    # Inplace restriction
    store.restrict_to(0, [1, 2], inplace=True)
    assert store._data.equals(pd.DataFrame({'nodes': [1, 2], 'edges': [1, 2]}))
    store = IncidenceStore(data)  # Recreate initial store

    # Non-inplace restriction (returns new dataframe)
    restricted_df = store.restrict_to(0, [1, 2], inplace=False)
    assert not restricted_df.equals(store._data)  # Should be a new dataframe
    assert restricted_df.equals(pd.DataFrame({'nodes': [1, 2], 'edges': [1, 2]}))

    # Invalid level
    with pytest.raises(ValueError):
        store.restrict_to(3, [1])  # Invalid level should raise error

    # Non-existent items
    store = IncidenceStore(data)
    restricted_df = store.restrict_to(0, [5], inplace=False)
    assert restricted_df.empty  # Empty dataframe as no pairs with item 5
