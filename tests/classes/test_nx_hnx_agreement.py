def assert_are_same_sets(s1, s2):
    assert set(s1) == set(s2)


def assert_are_same_set_of_sets(s1, s2):
    assert_are_same_sets(map(frozenset, s1), map(frozenset, s2))


def test_len(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that the length of the hypergraph returns the number of nodes
    """
    assert len(nx_graph) == len(hnx_graph_from_nx_graph)


def test_number_of_edges(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that the number of edges are the same
    """
    assert len(nx_graph.edges) == len(hnx_graph_from_nx_graph.edges)


def test_is_iterable(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that the object itself is iterable, and returns the node set
    """
    assert_are_same_sets(nx_graph, hnx_graph_from_nx_graph)


def test_is_subscriptable(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that the graph is subscriptable, i.e., hnx_graph_from_nx_graph[v], returns the neighbors of v
    """
    for v in nx_graph.nodes():
        assert_are_same_sets(nx_graph[v], hnx_graph_from_nx_graph[v])


def test_neighbors(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that nx_graph.neighbors(v) returns neighbors of vertex v.
    This is the same as nx_graph[v] above.
    """
    for v in nx_graph.nodes():
        assert_are_same_sets(nx_graph[v], hnx_graph_from_nx_graph[v])


def test_edges_iter(nx_graph, hnx_graph_from_nx_graph):
    """
    Confirm that hnx_graph_from_nx_graph.edges returns the same edges as nx_graph.edges()
    """
    assert_are_same_set_of_sets(
        nx_graph.edges(), hnx_graph_from_nx_graph.edges.elements.values()
    )
