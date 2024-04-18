from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
from hypernetx.classes.hypergraph import Hypergraph

from networkx.algorithms import bipartite

from hypernetx.classes.property_store import PropertyStore
from tests.classes.conftest import SevenBySix


#################Tests on constructors and from_<data type> ################################


@pytest.mark.parametrize(
    "hg",
    [
        Hypergraph(SevenBySix().edges_list),  # list of edges
        Hypergraph(SevenBySix().edgedict),  # dictionary of edges to nodes
        Hypergraph(SevenBySix().dataframe),  # dataframe of edges to nodes
    ],
)
def test_constructor_on_various_set_systems(hg):
    sbs = SevenBySix()
    assert len(hg.edges) == len(sbs.edges)
    assert len(hg.nodes) == len(sbs.nodes)

    assert hg.degree(sbs.nodes.A) == 3
    assert hg.order() == len(sbs.nodes)


@pytest.mark.parametrize("h", [Hypergraph(), Hypergraph({})])
def test_construct_empty_hypergraph(h):
    h = Hypergraph()
    assert h.shape == (0, 0)
    assert h.edges.is_empty()
    assert h.nodes.is_empty()


def test_from_incidence_dataframe(lesmis):
    h = Hypergraph(lesmis.edgedict)
    df = h.incidence_dataframe()
    hg = Hypergraph.from_incidence_dataframe(df)
    assert hg.shape == (40, 8)
    assert hg.size(3) == 8
    assert hg.degree("JA") == 3


def test_from_numpy_array(sbs):
    hg = Hypergraph.from_numpy_array(sbs.arr)
    assert len(hg.nodes) == len(sbs.arr)
    assert len(hg.edges) == len(sbs.arr[0])
    assert hg.dim("e5") == 2
    assert set(hg.neighbors("v2")) == {"v0", "v5"}


def test_from_bipartite(sbs):
    hg = Hypergraph(sbs.edgedict)
    hg_b = Hypergraph.from_bipartite(hg.bipartite())
    assert len(hg_b.edges) == len(sbs.edges)
    assert len(hg_b.nodes) == 7


#################tests on methods ################################
def test_len(sbs):
    hg = Hypergraph(sbs.edgedict)
    assert len(hg) == len(sbs.nodes)


def test_contains(sbs):
    hg = Hypergraph(sbs.edgedict)
    assert sbs.nodes.A in hg


def test_iterator(sbs):
    hg = Hypergraph(sbs.edgedict)
    nodes = [key for key in hg]
    assert sorted(nodes) == list(sbs.nodes)


def test_getitem(sbs):
    hg = Hypergraph(sbs.edgedict)
    nodes = hg[sbs.nodes.C]
    assert sorted(nodes) == [sbs.nodes.A, sbs.nodes.E, sbs.nodes.K]


def test_get_linegraph(sbs):
    hg = Hypergraph(sbs.edgedict)
    assert len(hg.edges) == len(sbs.edges)
    assert len(hg.nodes) == len(sbs.nodes)

    lg = hg.get_linegraph(s=1)

    diff = set(lg).difference(set(sbs.edges))
    assert len(diff) == 0


def test_add_edge_no_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_edge_with_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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
def test_add_edges_from_on_list_of_single_edge(sbs, new_edge, data, expected_props):
    h = Hypergraph(sbs.edgedict)
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
def test_add_edges_from_on_list_of_multiple_edges(sbs, edges):
    h = Hypergraph(sbs.edgedict)
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


def test_add_edges_from_on_list_of_edges_no_data(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_edges_from_on_list_of_edges_with_and_without_data(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_node_no_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_node_with_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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
def test_add_nodes_from_on_list_of_single_node(sbs, new_node, data, expected_props):
    h = Hypergraph(sbs.edgedict)
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
def test_add_nodes_from_on_list_of_multiple_nodes(sbs, nodes):
    h = Hypergraph(sbs.edgedict)
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


def test_add_nodes_from_on_list_of_nodes_no_data(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_nodes_from_on_list_of_nodes_with_and_without_data(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_incidence_on_existing_edges_node_no_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_incidence_user_on_existing_edges_node_with_attr(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_add_incidence_on_new_edge_new_node(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_remove_edges(sbs):
    hg = Hypergraph(sbs.edgedict)
    # shape returns (#nodes, #edges)
    assert hg.shape == (len(sbs.nodes), len(sbs.edges))

    # remove an edge containing nodes that are in other edges
    # the number of nodes should not decrease
    hg = hg.remove_edges(sbs.edges.P)
    assert hg.shape == (len(sbs.nodes), len(sbs.edges) - 1)

    # remove an edge containing a singleton ear (i.e. a node not present in other edges)
    # the number of nodes should decrease by exactly one
    hg = hg.remove_edges(sbs.edges.O)
    assert hg.shape == (len(sbs.nodes) - 1, len(sbs.edges) - 2)


def test_remove_nodes(sbs):
    hg = Hypergraph(sbs.edgedict)

    assert sbs.nodes.A in hg.nodes
    assert sbs.nodes.A in hg.edges[sbs.edges.P]
    assert sbs.nodes.A in hg.edges[sbs.edges.R]
    assert sbs.nodes.A in hg.edges[sbs.edges.S]

    hg_new = hg.remove_nodes(sbs.nodes.A)

    assert sbs.nodes.A not in hg_new.nodes
    assert sbs.nodes.A not in hg_new.edges[sbs.edges.P]
    assert sbs.nodes.A not in hg_new.edges[sbs.edges.R]
    assert sbs.nodes.A not in hg_new.edges[sbs.edges.S]


def test_remove_edges_nodes_incidences(triloop2):
    H = Hypergraph(triloop2.edgedict, name=triloop2.name)

    assert H.shape == (5, 4)
    duplicate_edge = ["ACD2"]
    newH = H.remove_edges(duplicate_edge)
    assert newH.shape == (5, 3)

    assert H.shape == (5, 4)
    newH = H.remove_nodes(["E"])
    assert newH.shape == (4, 4)

    assert H.shape == (5, 4)
    newH = H.remove_edges(["ACD"])
    assert newH.shape == (5, 3)

    # remove incidence in which the node is associated with other edges
    assert H.shape == (5, 4)
    newH = H.remove_incidences([("ACD", "E")])
    assert newH.shape == (5, 4)

    # removing 2 edges that have the node B. Node B is not associated with any other edge
    # new hypergraph should have 4 nodes and 2 edges
    assert H.shape == (5, 4)
    newH = H.remove_edges(["AB", "BC"])
    assert newH.shape == (4, 2)


def test_matrix(sbs):
    hg = Hypergraph(sbs.edgedict)

    assert hg.incidence_matrix().todense().shape == (len(sbs.nodes), len(sbs.edges))
    assert hg.adjacency_matrix(s=2).todense().shape == (7, 7)
    assert hg.edge_adjacency_matrix().todense().shape == (6, 6)

    aux_matrix = hg.auxiliary_matrix(node=False)
    assert aux_matrix.todense().shape == (6, 6)


def test_collapse_edges(sbs_dupes):
    hg = Hypergraph(sbs_dupes.edgedict)
    assert len(hg.edges) == len(sbs_dupes.edges)

    hc = hg.collapse_edges()
    assert len(hc.edges) == len(sbs_dupes.edges) - 1


def test_collapse_nodes(sbs_dupes):
    hg = Hypergraph(sbs_dupes.edgedict)
    assert len(hg.nodes) == len(sbs_dupes.nodes)

    hc = hg.collapse_nodes()
    assert len(hc.nodes) == len(sbs_dupes.nodes) - 1


def test_collapse_nodes_and_edges(sbs_dupes):
    hg = Hypergraph(sbs_dupes.edgedict)
    hc2 = hg.collapse_nodes_and_edges()

    assert len(hg.edges) == len(sbs_dupes.edges)
    assert len(hc2.edges) == len(sbs_dupes.edges) - 1
    assert len(hg.nodes) == len(sbs_dupes.nodes)
    assert len(hc2.nodes) == len(sbs_dupes.nodes) - 1


def test_restrict_to_edges(sbs):
    H = Hypergraph(sbs.edgedict)
    HS = H.restrict_to_edges(["P", "O"])
    assert len(H.edges) == 6
    assert len(HS.edges) == 2


def test_restrict_to_nodes(sbs):
    H = Hypergraph(sbs.edgedict)
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


def test_toplexes(sbs_dupes):
    h = Hypergraph(sbs_dupes.edgedict)
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
    E = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(E)
    assert h.shape == (9, 5)
    h1 = h.remove_singletons()
    assert h1.shape == (7, 3)
    assert h.shape == (9, 5)


def test_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    # h.components() causes an error
    assert [len(g) for g in h.component_subgraphs()] == [8]


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


def test_s_connected_components():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert list(h.s_connected_components()) == [{0, 1, 2, 3}]
    assert list(h.s_connected_components(s=2)) == [{1, 2, 3}]
    assert list(h.s_connected_components(s=2, edges=False)) == [{5, 6}]


def test_s_component_subgraphs():
    setsystem = [{1, 2, 3, 4}, {4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert {5, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=2, return_singletons=True)]
    )
    assert {3, 4}.issubset(
        [len(g) for g in h.s_component_subgraphs(s=3, return_singletons=True)]
    )


def test_size(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.size(sbs.edges.S) == len(sbs.edgedict[sbs.edges.S])
    assert h.size("S", {"T2", "V"}) == 2
    assert h.size("S", {"T1", "T2"}) == 1
    assert h.size("S", {"T2"}) == 1
    assert h.size("S", {"T1"}) == 0
    assert h.size("S", {}) == 0


def test_diameter(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.diameter() == 3


def test_diameter_should_raise_error(sbs):
    h = Hypergraph(sbs.edgedict)
    with pytest.raises(Exception) as excinfo:
        h.diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_node_diameters(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.node_diameters()[0] == 3
    assert h.node_diameters()[2] == [set(sbs.nodes)]


def test_edge_diameter(sbs):
    h = Hypergraph(sbs.edgedict)
    assert h.edge_diameter() == 3
    assert h.edge_diameters()[2] == [{"I", "L", "O", "P", "R", "S"}]
    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_bipartite(sbs):
    hg = Hypergraph(sbs.edgedict)
    assert bipartite.is_bipartite(hg.bipartite())


def test_dual(sbs):
    h = Hypergraph(sbs.edgedict)
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


def test_difference_on_empty_hypergraph(sbs):
    hg_empty = Hypergraph()
    hg = Hypergraph(sbs.edgedict)
    hg_diff = hg - hg_empty

    assert len(hg_diff) == len(sbs.nodes)
    assert len(hg_diff.nodes) == len(sbs.nodes)
    assert len(hg_diff.edges) == len(sbs.edges)
    assert hg_diff.shape == (len(sbs.nodes), len(sbs.edges))

    assert all(e in sbs.edges for e in hg_diff.edges)
    assert all(n in sbs.nodes for n in hg_diff.nodes)


def test_difference_on_similar_hypergraph(sbs):
    hg = Hypergraph(sbs.edgedict)

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
