from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import scipy

from networkx.algorithms import bipartite

from hypernetx.classes.hyp_view import HypergraphView
from hypernetx.classes.property_store import PropertyStore
from hypernetx.classes.hypergraph import Hypergraph


#################Tests on constructors and from_<data type> ################################


def test_constructor_on_list_of_edges(sevenbysix):
    hg = Hypergraph(sevenbysix.edges_list)
    assert len(hg.edges) == len(sevenbysix.edges)
    assert len(hg.nodes) == len(sevenbysix.nodes)

    assert hg.degree(sevenbysix.nodes.A) == 3
    assert hg.order() == len(sevenbysix.nodes)


def test_constructor_on_dict_of_edges_to_nodes(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    assert len(hg.edges) == len(sevenbysix.edges)
    assert len(hg.nodes) == len(sevenbysix.nodes)

    assert hg.degree(sevenbysix.nodes.A) == 3
    assert hg.order() == len(sevenbysix.nodes)


def test_constructor_on_dataframe(sevenbysix):
    hg = Hypergraph(sevenbysix.dataframe)
    assert len(hg.edges) == len(sevenbysix.edges)
    assert len(hg.nodes) == len(sevenbysix.nodes)

    assert hg.degree(sevenbysix.nodes.A) == 3
    assert hg.order() == len(sevenbysix.nodes)


@pytest.mark.parametrize("h", [Hypergraph(), Hypergraph({})])
def test_construct_empty_hypergraph(h):
    h = Hypergraph()
    assert h.shape == (0, 0)
    assert h.edges.is_empty()
    assert h.nodes.is_empty()


def test_from_incidence_dataframe_on_lesmis(lesmis):
    h = Hypergraph(lesmis.edgedict)
    df = h.incidence_dataframe()
    hg = Hypergraph.from_incidence_dataframe(df)
    assert hg.shape == (40, 8)
    assert hg.size(3) == 8
    assert hg.degree("JA") == 3


def test_from_incidence_dataframe():
    matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]])
    index = ["A", "B", "C"]
    columns = ["a", "b", "c", "d"]
    df = pd.DataFrame(matrix, index=index, columns=columns)

    h = Hypergraph.from_incidence_dataframe(df)

    assert "b" in h.edges()
    assert "d" not in h.edges()
    assert "C" in h.edges["a"]


def test_from_incidence_dataframe_with_key():
    matrix = np.array([[5, 0, 7, 2], [6, 8, 1, 1], [2, 5, 1, 9]])
    index = ["A", "B", "C"]
    columns = ["a", "b", "c", "d"]
    df = pd.DataFrame(matrix, index=index, columns=columns)

    h = Hypergraph.from_incidence_dataframe(df, key=lambda x: x > 4)

    assert "A" in h.edges["a"]
    assert "C" not in h.edges["a"]


def test_from_incidence_dataframe_with_fillna(sample_df):
    h = Hypergraph.from_incidence_dataframe(sample_df, fillna=1)
    assert "A" in h.edges["b"]


def test_from_numpy_array_on_sbs(sevenbysix):
    hg = Hypergraph.from_numpy_array(sevenbysix.arr)

    assert len(hg.nodes) == len(sevenbysix.arr)
    assert len(hg.edges) == len(sevenbysix.arr[0])
    assert hg.dim("e5") == 2
    assert set(hg.neighbors("v2")) == {"v0", "v5"}


def test_from_numpy_array_with_node_edge_names():
    matrix = np.array(
        [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    )
    node_names = ["A", "B", "C", "D"]
    edge_names = ["a", "b", "c", "d", "e"]

    h = Hypergraph.from_numpy_array(
        matrix, node_names=node_names, edge_names=edge_names
    )

    assert "a" in h.edges()
    assert "A" in h.nodes()
    assert "B" in h.edges["a"]


def test_from_numpy_array_with_key():
    """
    matrix should look like the following:
        a  b  c  d
     A  5  0  7  2
     B  6  8  1  1
     C  2  5  1  9

     With key=lambda x: x > 4, the Hypergraph will only consider valid incidences to have values greater than 4
    """
    matrix = np.array([[5, 0, 7, 2], [6, 8, 1, 1], [2, 5, 1, 9]])
    h = Hypergraph.from_numpy_array(
        matrix,
        node_names=["A", "B", "C"],
        edge_names=["a", "b", "c", "d"],
        key=lambda x: x > 4,
    )
    assert "A" in h.edges["a"]
    assert "C" not in h.edges["a"]


def test_from_bipartite_on_hnx_bipartite(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    hg_b = Hypergraph.from_bipartite(hg.bipartite())
    assert len(hg_b.edges) == len(sevenbysix.edges)
    assert len(hg_b.nodes) == 7


def test_from_bipartite_on_complete_bipartite():
    g = nx.complete_bipartite_graph(2, 3)
    left, right = nx.bipartite.sets(g)
    h = Hypergraph.from_bipartite(g)
    nodes = {*h.nodes}
    edges = {*h.edges}
    assert left.issubset(edges)
    assert right.issubset(nodes)

    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=4)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


#################tests on methods ################################
def test_len(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    assert len(hg) == len(sevenbysix.nodes)


def test_contains(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    assert sevenbysix.nodes.A in hg


def test_iterator(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    nodes = [key for key in hg]
    assert sorted(nodes) == list(sevenbysix.nodes)


def test_getitem(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    nodes = hg[sevenbysix.nodes.C]
    assert sorted(nodes) == [sevenbysix.nodes.A, sevenbysix.nodes.E, sevenbysix.nodes.K]


def test_get_linegraph(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    assert len(hg.edges) == len(sevenbysix.edges)
    assert len(hg.nodes) == len(sevenbysix.nodes)

    lg = hg.get_linegraph(s=1)

    diff = set(lg).difference(set(sevenbysix.edges))
    assert len(diff) == 0


def test_add_edge_no_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_edge = "X"

    assert h.shape == (7, 6)
    assert new_edge not in h.edges

    new_hg = h.add_edge(new_edge)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    assert new_hg.shape == (7, 6)
    assert new_edge not in new_hg.edges
    assert new_edge not in new_hg.incidences

    # The new edge will be saved in PropertyStore
    assert new_edge in new_hg.edges.property_store
    assert new_hg.edges.property_store[new_edge] == {"weight": 1}


def test_add_edge_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_edge = "X"

    assert h.shape == (7, 6)
    assert new_edge not in list(h.edges)

    new_hg = h.add_edge(new_edge, hair_color="red", age=42)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    assert new_hg.shape == (7, 6)
    assert new_edge not in new_hg.edges
    assert new_edge not in new_hg.incidences

    # The new edge will be saved in PropertyStore
    assert new_edge in new_hg.edges.property_store
    assert new_hg.edges.property_store[new_edge] == {
        "hair_color": "red",
        "age": 42,
        "weight": 1,
    }


@pytest.mark.parametrize(
    "new_edge, data, expected_props",
    [
        ("X", {}, {"weight": 1}),
        (
            "X",
            {"hair_color": "orange", "age": 42},
            {"hair_color": "orange", "age": 42, "weight": 1},
        ),
    ],
)
def test_add_edges_from_on_list_of_single_edge(
    sevenbysix, new_edge, data, expected_props
):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.shape == (7, 6)
    assert new_edge not in h.edges

    new_hg = h.add_edges_from([(new_edge, data)])

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    assert new_hg.shape == (7, 6)
    assert new_edge not in new_hg.edges
    assert new_edge not in new_hg.incidences

    # The new edge will be saved in PropertyStore
    assert new_edge in new_hg.edges.property_store
    assert new_hg.edges.property_store[new_edge] == expected_props


@pytest.mark.parametrize(
    "edges",
    [
        ([("X", {}), ("Y", {})]),
        ([("X", {"hair_color": "red"}), ("Y", {"age": "42"})]),
    ],
)
def test_add_edges_from_on_list_of_multiple_edges(sevenbysix, edges):
    h = Hypergraph(sevenbysix.edgedict)
    new_edges = [edge[0] for edge in edges]

    assert h.shape == (7, 6)
    assert all(new_edge not in h.edges for new_edge in new_edges)

    new_hg = h.add_edges_from(edges)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    assert new_hg.shape == (7, 6)
    assert all(new_edge not in new_hg.edges for new_edge in new_edges)
    assert all(new_edge not in new_hg.incidences for new_edge in new_edges)

    # The new edge will be saved in PropertyStore
    assert all(new_edge in new_hg.edges.property_store for new_edge in new_edges)
    for edge, expected_props in edges:
        # add the default weight to the expected properties
        expected_props.update({"weight": 1})
        assert new_hg.edges.property_store[edge] == expected_props


def test_add_edges_from_on_list_of_edges_no_data(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_edges = ["X", "Y"]

    assert h.shape == (7, 6)
    assert all(new_edge not in h.edges for new_edge in new_edges)

    new_hg = h.add_edges_from(new_edges)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    assert new_hg.shape == (7, 6)
    assert all(new_edge not in new_hg.edges for new_edge in new_edges)
    assert all(new_edge not in new_hg.incidences for new_edge in new_edges)

    # The new edge will be saved in PropertyStore
    assert all(new_edge in new_hg.edges.property_store for new_edge in new_edges)
    assert all(
        {"weight": 1} == new_hg.edges.property_store[new_edge] for new_edge in new_edges
    )


def test_add_edges_from_on_list_of_edges_with_and_without_data(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_edge1 = "X"
    new_edge2 = "Y"
    new_edge3 = "Z"
    new_edges = [new_edge1, new_edge2, new_edge3]

    assert h.shape == (7, 6)
    assert all(new_edge not in h.edges for new_edge in new_edges)

    new_hg = h.add_edges_from(
        [new_edge1, (new_edge2, {}), (new_edge3, {"hair_color": "red"})]
    )

    assert new_hg.shape == (7, 6)
    assert all(new_edge not in new_hg.edges for new_edge in new_edges)

    # The new edge will be saved in PropertyStore
    assert all(new_edge in new_hg.edges.property_store for new_edge in new_edges)

    assert new_hg.edges.property_store[new_edge1] == {"weight": 1}
    assert new_hg.edges.property_store[new_edge2] == {"weight": 1}
    assert new_hg.edges.property_store[new_edge3] == {"weight": 1, "hair_color": "red"}


def test_add_node_no_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_node = "Y"

    assert h.shape == (7, 6)
    assert new_node not in h.nodes

    new_hg = h.add_node(new_node)

    # the Hypergraph should not increase its number of nodes because the added node is not associated with any edge
    assert new_hg.shape == (7, 6)
    assert new_node not in new_hg.nodes

    # Check PropertyStore
    assert new_node in new_hg.nodes.property_store
    assert new_hg.nodes.property_store[new_node] == {"weight": 1}


def test_add_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_node = "Y"

    assert h.shape == (7, 6)
    assert new_node not in h.nodes

    new_hg = h.add_node(new_node, hair_color="red", age=42)

    # the Hypergraph should not increase its number of nodes because the added node is not associated with any edge
    assert new_hg.shape == (7, 6)
    assert new_node not in new_hg.nodes

    # Check PropertyStore
    assert new_node in new_hg.nodes.property_store
    assert new_hg.nodes.property_store[new_node] == {
        "hair_color": "red",
        "age": 42,
        "weight": 1,
    }


@pytest.mark.parametrize(
    "new_node, data, expected_props",
    [
        ("B", {}, {"weight": 1}),
        (
            "B",
            {"hair_color": "orange", "age": 42},
            {"hair_color": "orange", "age": 42, "weight": 1},
        ),
    ],
)
def test_add_nodes_from_on_list_of_single_node(
    sevenbysix, new_node, data, expected_props
):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.shape == (7, 6)
    assert new_node not in h.nodes

    new_hg = h.add_nodes_from([(new_node, data)])

    assert new_hg.shape == (7, 6)
    assert new_node not in new_hg.nodes

    # The new node will be saved in PropertyStore
    assert new_node in new_hg.nodes.property_store
    assert new_hg.nodes.property_store[new_node] == expected_props


@pytest.mark.parametrize(
    "nodes",
    [
        ([("B", {}), ("D", {})]),
        ([("B", {"hair_color": "red"}), ("D", {"age": "42"})]),
    ],
)
def test_add_nodes_from_on_list_of_multiple_nodes(sevenbysix, nodes):
    h = Hypergraph(sevenbysix.edgedict)
    new_nodes = [node[0] for node in nodes]

    assert h.shape == (7, 6)
    assert all(new_node not in h.nodes for new_node in new_nodes)

    new_hg = h.add_nodes_from(nodes)

    assert new_hg.shape == (7, 6)
    assert all(new_node not in new_hg.nodes for new_node in new_nodes)

    assert all(new_node in new_hg.nodes.property_store for new_node in new_nodes)
    for node, expected_props in nodes:
        # add the default weight to the expected properties
        expected_props.update({"weight": 1})
        assert new_hg.nodes.property_store[node] == expected_props


def test_add_nodes_from_on_list_of_nodes_no_data(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_nodes = ["B", "D"]

    assert h.shape == (7, 6)
    assert all(new_node not in h.nodes for new_node in new_nodes)

    new_hg = h.add_nodes_from(new_nodes)

    assert new_hg.shape == (7, 6)
    assert all(new_node not in new_hg.nodes for new_node in new_nodes)

    assert all(new_node in new_hg.nodes.property_store for new_node in new_nodes)
    assert all(
        {"weight": 1} == new_hg.nodes.property_store[new_node] for new_node in new_nodes
    )


def test_add_nodes_from_on_list_of_nodes_with_and_without_data(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    new_node1 = "B"
    new_node2 = "D"
    new_node3 = "F"
    new_nodes = [new_node1, new_node2, new_node3]

    assert h.shape == (7, 6)
    assert all(new_node not in h.nodes for new_node in new_nodes)

    new_hg = h.add_nodes_from(
        [new_node1, (new_node2, {}), (new_node3, {"hair_color": "red"})]
    )

    assert new_hg.shape == (7, 6)
    assert all(new_node not in new_hg.nodes for new_node in new_nodes)

    # The new edge will be saved in PropertyStore
    assert all(new_node in new_hg.nodes.property_store for new_node in new_nodes)

    assert new_hg.nodes.property_store[new_node1] == {"weight": 1}
    assert new_hg.nodes.property_store[new_node2] == {"weight": 1}
    assert new_hg.nodes.property_store[new_node3] == {"weight": 1, "hair_color": "red"}


def test_add_incidence_on_existing_edges_node_no_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "P"
    node = "E"
    new_incidence = (edge, node)

    assert edge in h.edges
    assert node in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    new_hg = h.add_incidence(edge, node)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in new_hg.incidences

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {"weight": 1}


def test_add_incidence_user_on_existing_edges_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "P"
    node = "E"
    new_incidence = (edge, node)

    assert edge in h.edges
    assert node in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    new_hg = h.add_incidence(edge, node, hair_color="red")

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in new_hg.incidences

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_incidence_on_new_edge_new_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "X"
    node = "B"
    new_incidence = (edge, node)

    assert edge not in h.edges
    assert node not in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    new_hg = h.add_incidence(edge, node, hair_color="red")

    # the Hypergraph should increase its number of edges and nodes because a new incidence was added
    assert new_hg.shape == (8, 7)
    assert new_incidence in new_hg.incidences
    assert edge in new_hg.edges
    assert node in new_hg.nodes

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_incidences_from_on_existing_edges_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "P"
    node = "E"
    new_incidence = (edge, node)

    assert edge in h.edges
    assert node in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    incidences = [(edge, node, {"hair_color": "red"})]
    new_hg = h.add_incidences_from(incidences)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in new_hg.incidences

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_incidences_from_on_new_edge_new_node(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)

    edge = "X"
    node = "B"
    new_incidence = (edge, node)

    assert edge not in h.edges
    assert node not in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    incidences = [(edge, node)]
    new_hg = h.add_incidences_from(incidences)

    # the Hypergraph should increase its number of edges and nodes because a new incidence was added
    assert new_hg.shape == (8, 7)
    assert new_incidence in new_hg.incidences
    assert edge in new_hg.edges
    assert node in new_hg.nodes

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
    }


def test_add_incidences_from_on_new_edge_new_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)

    edge = "X"
    node = "B"
    new_incidence = (edge, node)

    assert edge not in h.edges
    assert node not in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    incidences = [(edge, node, {"hair_color": "red"})]
    new_hg = h.add_incidences_from(incidences)

    # the Hypergraph should increase its number of edges and nodes because a new incidence was added
    assert new_hg.shape == (8, 7)
    assert new_incidence in new_hg.incidences
    assert edge in new_hg.edges
    assert node in new_hg.nodes

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_nodes_to_edges_on_existing_nodes_edges_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "P"
    node = "E"
    new_incidence = (edge, node)

    assert edge in h.edges
    assert node in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    edge_dict = {edge: {node: {"hair_color": "red"}}}
    new_hg = h.add_nodes_to_edges(edge_dict)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in new_hg.incidences

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_nodes_to_edges_on_existing_nodes_edges(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    edge = "P"
    node = "E"
    new_incidence = (edge, node)

    assert edge in h.edges
    assert node in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    edge_dict = {edge: [node]}
    new_hg = h.add_nodes_to_edges(edge_dict)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in new_hg.incidences

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {"weight": 1}


def test_add_nodes_to_edges_on_new_edge_new_node_with_attr(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)

    edge = "X"
    node = "B"
    new_incidence = (edge, node)

    assert edge not in h.edges
    assert node not in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    edge_dict = {edge: {node: {"hair_color": "red"}}}
    new_hg = h.add_nodes_to_edges(edge_dict)

    # the Hypergraph should increase its number of edges and nodes because a new incidence was added
    assert new_hg.shape == (8, 7)
    assert new_incidence in new_hg.incidences
    assert edge in new_hg.edges
    assert node in new_hg.nodes

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {
        "weight": 1,
        "hair_color": "red",
    }


def test_add_nodes_to_edges_on_new_edge_new_node(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)

    edge = "X"
    node = "B"
    new_incidence = (edge, node)

    assert edge not in h.edges
    assert node not in h.nodes
    assert h.shape == (7, 6)
    assert new_incidence not in h.incidences

    edge_dict = {edge: [node]}
    new_hg = h.add_nodes_to_edges(edge_dict)

    # the Hypergraph should increase its number of edges and nodes because a new incidence was added
    assert new_hg.shape == (8, 7)
    assert new_incidence in new_hg.incidences
    assert edge in new_hg.edges
    assert node in new_hg.nodes

    # Check PropertyStore
    assert new_incidence in new_hg.incidences.property_store
    assert new_hg.incidences.property_store[new_incidence] == {"weight": 1}


def test_clone(sevenbysix):
    original_hg = Hypergraph(sevenbysix.edgedict)
    cloned_hg = original_hg.clone()
    assert original_hg == cloned_hg
    assert cloned_hg != HypergraphView(None, None)


@pytest.mark.parametrize("edges_to_remove", ["O", ["O"]])
def test_remove_edges_inplace(sevenbysix, edges_to_remove):
    hg = Hypergraph(sevenbysix.edgedict, name="sbs")
    original_hg_id = id(hg)
    original_hg_name = hg.name

    # shape returns (#nodes, #edges)
    assert hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges))

    # remove an edge containing nodes that are in other edges
    # the number of nodes should not decrease
    hg = hg.remove_edges(sevenbysix.edges.P)
    assert hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges) - 1)
    assert id(hg) == original_hg_id
    assert hg.name == original_hg_name

    # remove an edge containing a singleton ear (i.e. a node not present in other edges)
    # the number of nodes should decrease by exactly one
    hg = hg.remove_edges(edges_to_remove)
    assert hg.shape == (len(sevenbysix.nodes) - 1, len(sevenbysix.edges) - 2)
    assert id(hg) == original_hg_id
    assert hg.name == original_hg_name


@pytest.mark.parametrize("edges_to_remove", ["O", ["O"]])
def test_remove_edges_not_inplace(sevenbysix, edges_to_remove):
    hg = Hypergraph(sevenbysix.edgedict, name="sbs")
    original_hg_id = id(hg)
    original_hg_name = hg.name
    new_name = "sbs_new"

    # shape returns (#nodes, #edges)
    assert hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges))

    # remove an edge containing nodes that are in other edges
    # the number of nodes should not decrease, number of edges should decrease by exactly one
    hg = hg.remove_edges(sevenbysix.edges.P, name=new_name, inplace=False)
    assert hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges) - 1)
    assert id(hg) != original_hg_id
    assert original_hg_name != hg.name
    assert hg.name == new_name

    # remove an edge containing a singleton ear (i.e. a node not present in other edges)
    # the number of nodes should decrease by exactly one, number of edges should decrease by exactly two
    hg = hg.remove_edges(edges_to_remove, name=new_name, inplace=False)
    assert hg.shape == (len(sevenbysix.nodes) - 1, len(sevenbysix.edges) - 2)
    assert id(hg) != original_hg_id
    assert original_hg_name != hg.name
    assert hg.name == new_name


@pytest.mark.parametrize("nodes_to_remove", ["A", ["A"]])
def test_remove_nodes_inplace(sevenbysix, nodes_to_remove):
    hg = Hypergraph(sevenbysix.edgedict, name="sbs")
    original_hg_id = id(hg)
    original_hg_name = hg.name

    assert sevenbysix.nodes.A in hg.nodes
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.P]
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.R]
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.S]

    hg_new = hg.remove_nodes(nodes_to_remove)

    assert sevenbysix.nodes.A not in hg_new.nodes
    assert sevenbysix.nodes.A not in hg_new.edges[sevenbysix.edges.P]
    assert sevenbysix.nodes.A not in hg_new.edges[sevenbysix.edges.R]
    assert sevenbysix.nodes.A not in hg_new.edges[sevenbysix.edges.S]

    assert id(hg) == original_hg_id
    assert hg_new.name == original_hg_name


@pytest.mark.parametrize("nodes_to_remove", ["A", ["A"]])
def test_remove_nodes_not_inplace(sevenbysix, nodes_to_remove):
    hg = Hypergraph(sevenbysix.edgedict, name="sbs")
    original_hg_id = id(hg)
    original_hg_name = hg.name
    new_name = "sbs_new"

    assert sevenbysix.nodes.A in hg.nodes
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.P]
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.R]
    assert sevenbysix.nodes.A in hg.edges[sevenbysix.edges.S]

    hg = hg.remove_nodes(nodes_to_remove, name=new_name, inplace=False)

    assert sevenbysix.nodes.A not in hg.nodes
    assert sevenbysix.nodes.A not in hg.edges[sevenbysix.edges.P]
    assert sevenbysix.nodes.A not in hg.edges[sevenbysix.edges.R]
    assert sevenbysix.nodes.A not in hg.edges[sevenbysix.edges.S]

    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name


def test_remove_edges_nodes_incidences_inplace(triloop2):
    # triloop2 is a hypergraph of the following shape:
    # {AB: {A, B}, BC: {B, C}, ACD: {A, C, D, E}, ACD2: {A, C, D, E}}
    hg = Hypergraph(triloop2.edgedict, name=triloop2.name)
    assert hg.shape == (5, 4)
    original_hg_id = id(hg)

    # removing a duplicate edge should not affect the number of nodes
    duplicate_edge = ["ACD2"]
    hg = hg.remove_edges(duplicate_edge)
    assert hg.shape == (5, 3)
    assert id(hg) == original_hg_id

    # number of nodes should drop by 1
    hg = hg.remove_nodes(["E"])
    assert hg.shape == (4, 3)
    assert id(hg) == original_hg_id

    # number of edges should drop by 1
    hg = hg.remove_edges(["ACD"])
    assert hg.shape == (3, 2)
    assert id(hg) == original_hg_id

    # remove an incidence that no longer exists; no change
    hg = hg.remove_incidences([("ACD", "E")])
    assert hg.shape == (3, 2)
    assert id(hg) == original_hg_id

    # removing the last two remaining edges will remove all the nodes from the hypergraph
    hg = hg.remove_edges(["AB", "BC"])
    assert hg.shape == (0, 0)
    assert id(hg) == original_hg_id


def test_remove_edges_nodes_incidences_not_inplace(triloop2):
    # triloop2 is a hypergraph of the following shape:
    # {AB: {A, B}, BC: {B, C}, ACD: {A, C, D, E}, ACD2: {A, C, D, E}}
    hg = Hypergraph(triloop2.edgedict, name=triloop2.name)
    assert hg.shape == (5, 4)
    original_hg_id = id(hg)
    original_hg_name = hg.name
    new_name = f"{triloop2.name}_new"

    # removing a duplicate edge should not affect the number of nodes
    duplicate_edge = ["ACD2"]
    hg = hg.remove_edges(duplicate_edge, name=new_name, inplace=False)
    assert hg.shape == (5, 3)
    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name

    # number of nodes should drop by 1
    hg = hg.remove_nodes(["E"], name=new_name, inplace=False)
    assert hg.shape == (4, 3)
    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name

    # number of edges should drop by 1
    hg = hg.remove_edges(["ACD"], name=new_name, inplace=False)
    assert hg.shape == (3, 2)
    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name

    # remove an incidence that no longer exists; no change
    hg = hg.remove_incidences([("ACD", "E")], name=new_name, inplace=False)
    assert hg.shape == (3, 2)
    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name

    # removing the last two remaining edges will remove all the nodes from the hypergraph
    hg = hg.remove_edges(["AB", "BC"], name=new_name, inplace=False)
    assert hg.shape == (0, 0)
    assert id(hg) != original_hg_id
    assert hg.name != original_hg_name
    assert hg.name == new_name


@pytest.mark.parametrize(
    "keys, expected_shape",
    [
        (
            ("O", "T1"),
            (6, 6),
        ),  # T1 is the only node associated with an edge; nodes reduced by 1
        (
            ("R", "A"),
            (7, 6),
        ),  # Removing this incidence does not remove either edge or node; no change
        (
            [("L", "C"), ("L", "E")],
            (7, 5),
        ),  # Removes the complete hyperedge L; edges reduced by 1, nodes remain unchanged
    ],
)
def test_remove_incidences_inplace(sevenbysix, keys, expected_shape):
    hg = Hypergraph(sevenbysix.edgedict)
    original_hg_id = id(hg)

    assert hg.shape == (7, 6)

    hg = hg.remove_incidences(keys)

    assert hg.shape == expected_shape
    assert id(hg) == original_hg_id


@pytest.mark.parametrize(
    "keys, expected_shape",
    [
        (
            ("O", "T1"),
            (6, 6),
        ),  # T1 is the only node associated with an edge; nodes reduced by 1
        (
            ("R", "A"),
            (7, 6),
        ),  # Removing this incidence does not remove either edge or node; no change
        (
            [("L", "C"), ("L", "E")],
            (7, 5),
        ),  # Removes the complete hyperedge L; edges reduced by 1, nodes remain unchanged
    ],
)
def test_remove_incidences_not_inplace(sevenbysix, keys, expected_shape):
    hg = Hypergraph(sevenbysix.edgedict, name="sbs")
    original_hg_id = id(hg)
    original_name = hg.name
    new_name = "sbs_new"

    assert hg.shape == (7, 6)

    hg = hg.remove_incidences(keys, name=new_name, inplace=False)

    assert hg.shape == expected_shape
    assert id(hg) != original_hg_id
    assert hg.name != original_name
    assert hg.name == new_name


def test_incidence_matrix(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    incidence_matrix, nodes_idx, edges_idx = hg.incidence_matrix(index=True)

    # check the incidence matrix elements and shape
    assert incidence_matrix.todense().shape == (
        len(sevenbysix.nodes),
        len(sevenbysix.edges),
    )
    assert np.allclose(incidence_matrix.A, sevenbysix.incidence_matrix.A)

    # check the edge and node indexes
    assert nodes_idx.tolist() == list(sevenbysix.nodes)
    assert edges_idx.tolist() == list(sevenbysix.edges)


def test_adjacency_matrix(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    adjacency_matrix, node_idx = hg.adjacency_matrix(index=True)

    assert adjacency_matrix.todense().shape == (
        len(sevenbysix.nodes),
        len(sevenbysix.nodes),
    )
    assert np.allclose(adjacency_matrix.A, sevenbysix.s1_adjacency_matrx.A)
    assert node_idx.tolist() == list(sevenbysix.nodes)


def test_edge_adjacency_matrix(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    adjacency_matrix, node_idx = hg.edge_adjacency_matrix(index=True)

    assert adjacency_matrix.todense().shape == (
        len(sevenbysix.edges),
        len(sevenbysix.edges),
    )
    assert np.allclose(adjacency_matrix.A, sevenbysix.s1_edge_adjacency_matrx.A)
    assert node_idx.tolist() == list(sevenbysix.edges)


def test_auxiliary_matrix_on_nodes(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    aux_matrix, indexes = hg.auxiliary_matrix(index=True)

    assert aux_matrix.todense().shape == (len(sevenbysix.nodes), len(sevenbysix.nodes))
    assert np.allclose(aux_matrix.A, sevenbysix.s1_adjacency_matrx.A)
    assert indexes.tolist() == list(sevenbysix.nodes)


def test_auxiliary_matrix_on_edges(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    aux_matrix, indexes = hg.auxiliary_matrix(node=False, index=True)

    assert aux_matrix.todense().shape == (len(sevenbysix.edges), len(sevenbysix.edges))
    assert np.allclose(aux_matrix.A, sevenbysix.s1_edge_adjacency_matrx.A)
    assert indexes.tolist() == list(sevenbysix.edges)


def test_collapse_edges(sevenbysix_dupes):
    hg = Hypergraph(sevenbysix_dupes.edgedict)
    assert len(hg.edges) == len(sevenbysix_dupes.edges)

    hc = hg.collapse_edges()
    assert len(hc.edges) == len(sevenbysix_dupes.edges) - 1


def test_collapse_nodes(sevenbysix_dupes):
    hg = Hypergraph(sevenbysix_dupes.edgedict)
    assert len(hg.nodes) == len(sevenbysix_dupes.nodes)

    hc = hg.collapse_nodes()
    assert len(hc.nodes) == len(sevenbysix_dupes.nodes) - 1


def test_collapse_nodes_and_edges(sevenbysix_dupes):
    hg = Hypergraph(sevenbysix_dupes.edgedict)
    hc2 = hg.collapse_nodes_and_edges()

    assert len(hg.edges) == len(sevenbysix_dupes.edges)
    assert len(hc2.edges) == len(sevenbysix_dupes.edges) - 1
    assert len(hg.nodes) == len(sevenbysix_dupes.nodes)
    assert len(hc2.nodes) == len(sevenbysix_dupes.nodes) - 1


def test_restrict_to_edges(sevenbysix):
    H = Hypergraph(sevenbysix.edgedict)
    HS = H.restrict_to_edges(["P", "O"])
    assert len(H.edges) == 6
    assert len(HS.edges) == 2


def test_restrict_to_nodes(sevenbysix):
    H = Hypergraph(sevenbysix.edgedict)
    assert len(H.nodes) == 7
    H1 = H.restrict_to_nodes(["A", "E", "K"])
    assert len(H.nodes) == 7
    assert len(H1.nodes) == 3
    assert len(H1.edges) == 5
    assert "C" in H.edges["P"]
    assert "C" not in H1.edges["P"]


def test_remove_from_restriction(triloop):
    h = Hypergraph(triloop.edgedict)
    h1 = h.restrict_to_nodes(h.neighbors("A")).remove_nodes(
        "A"
    )  # Hypergraph does not have a remove_node method
    assert "A" not in h1
    assert "A" not in h1.edges["ACD"]


def test_toplexes(sevenbysix_dupes):
    h = Hypergraph(sevenbysix_dupes.edgedict)
    T = h.toplexes(return_hyp=True)
    assert len(T.nodes) == 8
    assert len(T.edges) == 5
    T = T.collapse_nodes()
    assert len(T.nodes) == 7


def test_is_connected():
    setsystem = [{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert h.is_connected() is True
    assert h.is_connected(s=2) is False
    assert h.is_connected(s=2, edges=True) is True
    # test case below will raise nx.NetworkXPointlessConcept
    assert h.is_connected(s=3, edges=True) is False


def test_singletons():
    E = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(E)
    assert h.shape == (9, 5)
    singles = h.singletons()
    assert len(singles) == 2
    h = h.remove_edges(singles)
    assert h.shape == (7, 3)


def test_remove_singletons():
    edges_dict = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(edges_dict)
    assert h.shape == (9, 5)

    h1 = h.remove_singletons()

    assert h1.shape == (7, 3)
    assert h.shape == (9, 5)
    assert id(h) != id(h1)


@pytest.mark.parametrize(
    "edges, expected_components",
    [(False, [{1, 2, 3, 4}, {5, 6}, {7}]), (True, [{"C", "D"}, {"A", "B"}, {"E"}])],
)
def test_components(edges, expected_components):
    setsystem = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {5, 6}, "D": {6}, "E": {7}}
    h = Hypergraph(setsystem)

    actual_components = list(h.components(edges=edges))

    assert all(expected_c in actual_components for expected_c in expected_components)


def test_connected_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert len(list(h.connected_components())) == 1
    assert list(h.connected_components(edges=True)) == [{0, 1, 2, 3}]
    assert [len(g) for g in h.connected_component_subgraphs()] == [8]


def test_s_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert len(list(h.s_components())) == 1
    assert len(list(h.s_components(s=2))) == 2
    assert len(list(h.s_components(s=3))) == 4
    assert len(list(h.s_components(s=3, edges=False))) == 7
    assert len(list(h.s_components(s=4, edges=False))) == 8


@pytest.mark.parametrize(
    "edges, return_singletons, expected_connected_components",
    [
        (True, False, [{"C", "D"}, {"A", "B"}]),
        (True, True, [{"C", "D"}, {"A", "B"}, {"E"}]),
        (False, False, [{1, 2, 3, 4}, {5, 6}]),
        (False, True, [{1, 2, 3, 4}, {5, 6}, {7}]),
    ],
)
def test_s_connected_components(
    edges, return_singletons, expected_connected_components
):
    setsystem = {"A": {1, 2, 3}, "B": {2, 3, 4}, "C": {5, 6}, "D": {6}, "E": {7}}
    h = Hypergraph(setsystem)

    actual_connected_components = list(
        h.s_connected_components(edges=edges, return_singletons=return_singletons)
    )

    assert all(
        expected_c in actual_connected_components
        for expected_c in expected_connected_components
    )


def test_s_component_subgraphs():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert {5, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=2, return_singletons=True)]
    )
    assert {3, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=3, return_singletons=True)]
    )


def test_size(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.size(sevenbysix.edges.S) == len(sevenbysix.edgedict[sevenbysix.edges.S])
    assert h.size("S", {"T2", "V"}) == 2
    assert h.size("S", {"T1", "T2"}) == 1
    assert h.size("S", {"T2"}) == 1
    assert h.size("S", {"T1"}) == 0
    assert h.size("S", {}) == 0


def test_diameter(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.diameter() == 3


def test_diameter_should_raise_error(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    with pytest.raises(Exception) as excinfo:
        h.diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_node_diameters(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.node_diameters()[0] == 3
    assert h.node_diameters()[2] == [set(sevenbysix.nodes)]


def test_edge_diameter(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    assert h.edge_diameter() == 3
    assert h.edge_diameters()[2] == [{"I", "L", "O", "P", "R", "S"}]
    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_bipartite(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    assert bipartite.is_bipartite(hg.bipartite())


def test_dual(sevenbysix):
    h = Hypergraph(sevenbysix.edgedict)
    hd = h.dual()
    assert isinstance(hd.nodes.property_store, PropertyStore)
    assert isinstance(hd.edges.property_store, PropertyStore)
    assert set(h.nodes) == set(hd.edges)
    assert set(h.edges) == set(hd.nodes)
    assert list(h.dataframe.columns) == list(hd.dataframe.columns)


@pytest.mark.filterwarnings("ignore:No 3-path between ME and FN")
def test_distance(lesmis):
    h = Hypergraph(lesmis.edgedict)
    assert h.distance("ME", "FN") == 2
    assert h.distance("ME", "FN", s=2) == 3
    assert h.distance("ME", "FN", s=3) == np.inf


def test_edge_distance(lesmis):
    h = Hypergraph(lesmis.edgedict)
    assert h.edge_distance(1, 4) == 2
    h2 = h.remove_incidences([5])
    assert h2.edge_distance(1, 4) == 3
    assert h2.edge_distance(1, 4, s=2) == np.inf


def test_incidence_dataframe(lesmis):
    h = Hypergraph(lesmis.edgedict)
    df = h.incidence_dataframe()
    assert np.allclose(np.array(np.sum(df)), np.array([10, 9, 8, 4, 8, 3, 12, 6]))


def test_static_hypergraph_s_connected_components(lesmis):
    h = Hypergraph(lesmis.edgedict)
    assert {7, 8} in list(h.s_connected_components(edges=True, s=4))


def test_difference_on_same_hypergraph(lesmis):
    hg = Hypergraph(lesmis.edgedict)
    hg_copy = Hypergraph(lesmis.edgedict)

    hg_diff = hg - hg_copy

    assert len(hg_diff) == 0
    assert len(hg_diff.nodes) == 0
    assert len(hg_diff.edges) == 0
    assert hg_diff.shape == (0, 0)
    assert hg_diff.incidence_dict == {}


def test_difference_on_empty_hypergraph(sevenbysix):
    hg_empty = Hypergraph()
    hg = Hypergraph(sevenbysix.edgedict)
    hg_diff = hg - hg_empty

    assert len(hg_diff) == len(sevenbysix.nodes)
    assert len(hg_diff.nodes) == len(sevenbysix.nodes)
    assert len(hg_diff.edges) == len(sevenbysix.edges)
    assert hg_diff.shape == (len(sevenbysix.nodes), len(sevenbysix.edges))

    assert all(e in sevenbysix.edges for e in hg_diff.edges)
    assert all(n in sevenbysix.nodes for n in hg_diff.nodes)


def test_difference_on_similar_hypergraph(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    # create a hypergraph based on hg, but remove the 'I' edge
    a, c, e, k, t1, t2, v = ("A", "C", "E", "K", "T1", "T2", "V")
    l, o, p, r, s = ("L", "O", "P", "R", "S")
    data = OrderedDict(
        [(p, {a, c, k}), (r, {a, e}), (s, {a, k, t2, v}), (l, {c, e}), (o, {t1, t2})]
    )
    hg_similar = Hypergraph(data, edge_col="edges", node_col="nodes")

    # returns a hypergraph with one edge and two nodes
    hg_diff = hg - hg_similar

    assert len(hg_diff) == 2
    assert len(hg_diff.nodes) == 2
    assert len(hg_diff.edges) == 1
    assert hg_diff.shape == (2, 1)

    edges_diff = ["I"]
    assert all(edge in edges_diff for edge in hg_diff.edges)

    nodes_diff = ["K", "T2"]
    assert all(node in nodes_diff for node in hg_diff.nodes)


def test_sum_hypergraph_empty_hypergraph(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)
    hg_to_add = Hypergraph()

    # add empty hypergraph to hypergraph
    new_hg = hg.sum(hg_to_add)
    assert new_hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges))

    # add hypergraph to empty hypergraph
    new_hg = hg_to_add.sum(hg)
    assert new_hg.shape == (len(sevenbysix.nodes), len(sevenbysix.edges))


def test_sum_hypergraph_with_dupe_hypergraph(sevenbysix, sevenbysix_dupes):
    hg = Hypergraph(sevenbysix.edgedict)
    hg_dupes = Hypergraph(sevenbysix_dupes.edgedict)

    # Case: HDuplicate + H
    # add almost duplicate hypergraph to hypergraph
    new_hg = hg.sum(hg_dupes)

    assert new_hg.shape == (len(sevenbysix.nodes) + 1, len(sevenbysix.edges) + 1)
    # check for new incidences
    expected_incidences = {
        "I": ["K", "T2"],
        "L": ["C", "E", "F"],
        "M": ["C", "E", "F"],
        "O": ["T1", "T2"],
        "P": ["A", "C", "K"],
        "R": ["A", "E", "F"],
        "S": ["A", "K", "T2", "V"],
    }
    actual_incidences = new_hg.incidence_dict
    for expected_edge, expected_nodes in expected_incidences.items():
        assert expected_edge in actual_incidences
        assert all(node in actual_incidences[expected_edge] for node in expected_nodes)

    # Case: H + HDuplicate
    # add hypergraph to almost duplicate
    new_hg = hg_dupes.sum(hg)

    assert new_hg.shape == (len(sevenbysix.nodes) + 1, len(sevenbysix.edges) + 1)
    # check for new incidences
    actual_incidences = new_hg.incidences.incidence_dict
    for expected_edge, expected_nodes in expected_incidences.items():
        assert expected_edge in actual_incidences
        assert all(node in actual_incidences[expected_edge] for node in expected_nodes)


@pytest.mark.parametrize(
    "edges, expected_edges",
    [
        ({"P": "Papa"}, ["Papa", "R", "S", "L", "O", "I"]),
        (
            {
                "P": "Papa",
                "R": "Romeo",
                "S": "Sierra",
                "L": "Lima",
                "O": "Oscar",
                "I": "India",
            },
            ["Papa", "Romeo", "Sierra", "Lima", "Oscar", "India"],
        ),
    ],
)
def test_rename_edges(sevenbysix, edges, expected_edges):
    hg = Hypergraph(sevenbysix.edgedict)
    assert all(e in hg.edges for e in sevenbysix.edges)

    new_hg = hg.rename(edges=edges)

    assert all(e in new_hg.edges for e in expected_edges)


@pytest.mark.parametrize(
    "nodes, expected_nodes",
    [
        ({"A": "Alpha"}, ["Alpha", "C", "E", "K", "T1", "T2", "V"]),
        (
            {
                "A": "Alpha",
                "C": "Charlie",
                "E": "Echo",
                "K": "Kilo",
                "T1": "Tango1",
                "T2": "Tango2",
                "V": "Victor",
            },
            ["Alpha", "Charlie", "Echo", "Kilo", "Tango1", "Tango2", "Victor"],
        ),
    ],
)
def test_rename_nodes(sevenbysix, nodes, expected_nodes):
    hg = Hypergraph(sevenbysix.edgedict)
    assert all(n in hg.nodes for n in sevenbysix.nodes)

    new_hg = hg.rename(nodes=nodes)

    assert all(n in new_hg.nodes for n in expected_nodes)


def test_rename_on_no_op(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    new_hg = hg.rename()

    assert id(hg) == id(new_hg)


@pytest.mark.parametrize(
    "prop_name, expected_property",
    [
        (None, {"weight": 1.0}),
        ("weight", 1),
        ("misc_properties", {}),
        ("not-a-prop-return-None", None),
    ],
)
def test_get_cell_properties(sevenbysix, prop_name, expected_property):
    hg = Hypergraph(sevenbysix.edgedict)

    prop = hg.get_cell_properties(
        sevenbysix.edges.P, sevenbysix.nodes.A, prop_name=prop_name
    )

    assert prop == expected_property


@pytest.mark.parametrize(
    "uid, level, prop_name, expected_property",
    [
        ("P", 0, None, {"weight": 1.0}),
        ("P", 0, "weight", 1.0),
        ("P", 0, "not-a-prop", None),
        ("A", 1, None, {"weight": 1.0}),
        ("A", 1, "weight", 1.0),
        ("A", 1, "not-a-prop", None),
        (("P", "A"), 2, None, {"weight": 1.0}),
        (("P", "A"), 2, "weight", 1.0),
        (("P", "A"), 2, "not-a-prop", None),
    ],
)
def test_get_properties(sevenbysix, uid, level, prop_name, expected_property):
    hg = Hypergraph(sevenbysix.edgedict)

    props = hg.get_properties(uid, level=level, prop_name=prop_name)

    assert props == expected_property


@pytest.mark.parametrize(
    "node_uid, s, expected_degree", [("A", 1, 3), ("A", 2, 3), ("A", 3, 2)]
)
def test_degree(sevenbysix, node_uid, s, expected_degree):
    hg = Hypergraph(sevenbysix.edgedict)

    assert hg.degree(node_uid, s=s) == expected_degree


@pytest.mark.parametrize(
    "node, s, expected_neighbors",
    [
        ("A", 1, ["C", "K", "E", "T2", "V"]),
        ("T2", 2, ["K"]),
    ],
)
def test_neighbors(sevenbysix, node, s, expected_neighbors):
    hg = Hypergraph(sevenbysix.edgedict)

    neighbors = hg.neighbors(node, s=s)

    assert all(n in neighbors for n in expected_neighbors)


def test_neighbors_on_invalid_node(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    neighbors = hg.neighbors("NEMO")

    assert neighbors == []


@pytest.mark.parametrize(
    "edge, s, expected_neighbors",
    [("P", 1, ["I", "L", "R", "S"]), ("O", 2, []), ("I", 2, ["S"])],
)
def test_edge_neighbors(sevenbysix, edge, s, expected_neighbors):
    hg = Hypergraph(sevenbysix.edgedict)

    neighbors = hg.edge_neighbors(edge, s=s)

    assert all(n in neighbors for n in expected_neighbors)


def test_neighbors_on_invalid_edge(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    neighbors = hg.edge_neighbors("NEMO")

    assert neighbors == []


def test_edge_size_dist(sevenbysix):
    hg = Hypergraph(sevenbysix.edgedict)

    edge_sizes = hg.edge_size_dist()

    expected_sizes = [2, 2, 2, 3, 2, 4]

    assert all(e_size in expected_sizes for e_size in edge_sizes)


@pytest.mark.parametrize(
    "edges, groupings",
    [
        (True, [["I"], ["L", "M"], ["O"], ["P"], ["R"], ["S"]]),
        (False, [["A"], ["C"], ["E", "F"], ["K"], ["T1"], ["T2"], ["V"]]),
    ],
)
def test_equivalence_classes(sevenbysix_dupes, edges, groupings):
    hg = Hypergraph(sevenbysix_dupes.edgedict)

    res = hg.equivalence_classes(edges=edges)

    assert res == groupings
