import pytest


def assert_are_same_sets(S, T):
    assert set(S) == set(T)


def assert_are_same_set_of_sets(S, T):
    assert_are_same_sets(map(frozenset, S), map(frozenset, T))


def test_len(G, H):
    """
    Confirm that the length of the hypergraph returns the number of nodes
    """
    assert len(G) == len(H)


def test_number_of_edges(G, H):
    """
    Confirm that the number of edges are the same
    """
    assert len(G.edges) == len(H.edges)


def test_is_iterable(G, H):
    """
    Confirm that the object itself is iterable, and returns the node set
    """
    assert_are_same_sets(G, H)


def test_is_subscriptable(G, H):
    """
    Confirm that the graph is subscriptable, i.e., G[v], returns the neighbors of v
    """
    for v in G.nodes():
        assert_are_same_sets(G[v], H[v])


def test_neighbors(G, H):
    """
    Confirm that G.neighbors(v) returns neighbors of vertex v.
    This is the same as G[v] above.
    """
    for v in G.nodes():
        assert_are_same_sets(G[v], H[v])


@pytest.mark.xfail(reason="Hypergraph edges do not match edges in nx graph")
def test_edges_iter(G, H):
    # breakpoint()
    assert_are_same_set_of_sets(G.edges(), H.edges())
