from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from hypernetx import Hypergraph, EntitySet



def test_from_bipartite():
    g = nx.complete_bipartite_graph(2, 3)
    left, right = nx.bipartite.sets(g)
    h = Hypergraph.from_bipartite(g)
    assert left.issubset(h.nodes)
    assert right.issubset(h.edges)
    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=4)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


@pytest.mark.parametrize("static", [(True), (False)])
def test_hypergraph_from_bipartite_and_from_constructor_should_be_equal(seven_by_six, static):
    edgedict = OrderedDict(seven_by_six.edgedict)

    bipartite_graph = Hypergraph(edgedict).bipartite()
    hg_from_bipartite = Hypergraph.from_bipartite(bipartite_graph, static=static)

    entityset = EntitySet("_", edgedict)
    hg_from_constructor = Hypergraph(entityset, static=static)

    assert hg_from_bipartite.isstatic == hg_from_constructor.isstatic

    assert hg_from_bipartite.shape == hg_from_constructor.shape

    incidence_dict_hg_from_bipartite = {key: sorted(value) for key, value in hg_from_bipartite.incidence_dict.items()}
    incidence_dict_hg_from_constructor = {key: sorted(value) for key, value in hg_from_constructor.incidence_dict.items()}
    assert incidence_dict_hg_from_bipartite == incidence_dict_hg_from_constructor


def test_from_numpy_array():
    M = np.array([[0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    h = Hypergraph.from_numpy_array(M)
    assert "v1" in h.edges["e0"]
    assert "e1" not in h.nodes["v2"].memberships
    with pytest.raises(Exception) as excinfo:
        h = Hypergraph.from_numpy_array(M, node_names=["A"])
    assert "Number of node names does not match number of rows" in str(excinfo.value)
    node_names = ["A", "B", "C", "D"]
    edge_names = ["a", "b", "c", "d", "e"]
    h = Hypergraph.from_numpy_array(M, node_names, edge_names)
    assert "a" in h.edges
    assert "A" in h.nodes
    assert "B" in h.edges["a"]


def test_from_numpy_array_with_key():
    M = np.array([[5, 0, 7, 2], [6, 8, 1, 1], [2, 5, 1, 9]])
    h = Hypergraph.from_numpy_array(
        M,
        node_names=["A", "B", "C"],
        edge_names=["a", "b", "c", "d"],
        key=lambda x: x > 4,
    )
    assert "A" in h.edges["a"]
    assert "C" not in h.edges["a"]


def test_from_dataframe():
    M = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]])
    index = ["A", "B", "C"]
    columns = ["a", "b", "c", "d"]
    df = pd.DataFrame(M, index=index, columns=columns)
    h = Hypergraph.from_dataframe(df)
    assert "b" in h.edges
    assert "d" not in h.edges
    assert "C" in h.edges["a"]


def test_from_dataframe_with_key():
    M = np.array([[5, 0, 7, 2], [6, 8, 1, 1], [2, 5, 1, 9]])
    index = ["A", "B", "C"]
    columns = ["a", "b", "c", "d"]
    df = pd.DataFrame(M, index=index, columns=columns)
    h = Hypergraph.from_dataframe(df, key=lambda x: x > 4)
    assert "A" in h.edges["a"]
    assert "C" not in h.edges["a"]


def test_from_dataframe_with_transforms_and_fillna(dataframe):
    df = dataframe.df

    def key1(x):
        return x ** 2

    def key2(x):
        return (x < 5) * x

    def key3(x):
        return (x > 0) * x

    h = Hypergraph.from_dataframe(df)
    assert "A" in h.edges["a"]
    assert "A" not in h.edges["b"]
    h = Hypergraph.from_dataframe(df, fillna=1)
    assert "A" in h.edges["b"]
    h = Hypergraph.from_dataframe(df, transforms=[key1, key2])
    assert "A" in h.edges["c"]
    assert "C" not in h.edges["b"]
    h = Hypergraph.from_dataframe(df, transforms=[key2, key3])
    assert "C" in h.edges["b"]
    h = Hypergraph.from_dataframe(df, transforms=[key3, key1], key=key2)
    assert "A" not in h.edges["a"]
    assert "B" in h.edges["b"]
    assert "C" not in h.edges["c"]
    assert "C" in h.edges["a"]
