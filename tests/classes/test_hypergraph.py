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
    assert hg.number_of_edges() == len(sbs.edges)
    assert hg.number_of_nodes() == len(sbs.nodes)

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


def test_add_edge_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    new_edge = "X"

    assert h.shape == (7, 6)
    assert new_edge not in list(h.edges)

    # add a new edge in place; i.e. the current hypergraph should be mutated
    h.add_edge(new_edge)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # an edge does not connect two or more nodes.
    # In other words, adding an edge with no nodes
    assert h.shape == (7, 6)
    assert new_edge not in list(h.edges)

    # the new edge has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_edge not in h.edges.properties

    # However, the new_edge will be listed in the complete list of all user and non-user-define properties for all edges
    assert new_edge in h.edges.to_dataframe.index.tolist()

    assert new_edge in h.edges.to_dataframe.index.tolist()


def test_add_edge_not_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    new_edge = "X"

    assert h.shape == (7, 6)
    assert new_edge not in list(h.edges)

    # add a new edge not in place; the current hypergraph should be diffrent from the new hypergraph
    # created from add_edge
    new_hg = h.add_edge(new_edge, inplace=False)

    assert new_hg.shape == (7, 6)
    assert new_edge not in list(new_hg.edges)

    assert new_edge not in new_hg.edges.properties
    assert new_edge in new_hg.edges.to_dataframe.index.tolist()

    # verify that the new edge is not in the old HyperGraph
    assert new_edge not in list(h.edges)


def test_add_node_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    new_node = "Y"

    assert h.shape == (7, 6)
    assert new_node not in list(h.nodes)

    h.add_node(new_node)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # a node does not add a node to the incidence store.
    # In other words, adding an edge with no nodes
    assert h.shape == (7, 6)
    assert new_node not in list(h.nodes)

    # the new node has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_node not in h.nodes.properties

    # However, the new node will be listed in the complete list of all user and non-user-define properties for all nodes
    assert new_node in h.nodes.to_dataframe.index.tolist()


def test_add_node_not_inplace(sbs):
    h = Hypergraph(sbs.edgedict)
    new_node = "Y"

    assert h.shape == (7, 6)
    assert new_node not in list(h.nodes)

    new_hg = h.add_node(new_node, inplace=False)

    # the Hypergraph should not increase its number of edges and incidences because the current behavior of adding
    # a node does not add a node to the incidence store.
    # In other words, adding an edge with no nodes
    assert new_hg.shape == (7, 6)
    assert new_node not in list(new_hg.nodes)

    # the new node has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_node not in new_hg.nodes.properties

    # However, the new node will be listed in the complete list of all user and non-user-define properties for all nodes
    assert new_node in new_hg.nodes.to_dataframe.index.tolist()

    # check that the new node is not in the old hypergraph
    assert new_node not in list(h.nodes)


def test_add_incidence_inplace(sbs):
    h = Hypergraph(sbs.edgedict)

    # Case 1: Add a new incidence using an existing edge and
    # exisiting node that are not currently associated with each other
    new_incidence = ("P", "E")

    assert h.shape == (7, 6)
    assert new_incidence not in list(h.incidences)

    h.add_incidence(new_incidence)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert h.shape == (7, 6)
    assert new_incidence in list(h.incidences)

    # the new incidence has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_incidence not in h.incidences.properties

    # However, the new incidence will be listed in the complete list of all user and non-user-define properties
    # for all incidences
    assert new_incidence in h.incidences.to_dataframe.index.tolist()


def test_add_incidence_not_inplace(sbs):
    h = Hypergraph(sbs.edgedict)

    # Case 1: Add a new incidence using an existing edge and
    # exisiting node that are not currently associated with each other
    new_incidence = ("P", "E")

    assert h.shape == (7, 6)
    assert new_incidence not in list(h.incidences)

    new_hg = h.add_incidence(new_incidence, inplace=False)

    # the Hypergraph should not increase its number of edges and nodes because no new edges or nodes were added
    assert new_hg.shape == (7, 6)
    assert new_incidence in list(new_hg.incidences)

    # the new incidence has no user-defined property data, so it should not be listed in the PropertyStore
    assert new_incidence not in new_hg.incidences.properties

    # However, the new incidence will be listed in the complete list of all user and non-user-define properties
    # for all incidences
    assert new_incidence in new_hg.incidences.to_dataframe.index.tolist()

    # check that node E has edge P in its memberships
    # assert new_incidence[0] in h.incidences.memberships[new_incidence[1]]

    # also check that the incidence is not in the old hypergraph
    assert new_incidence not in list(h.incidences)


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


def test_remove(triloop2):
    H = Hypergraph(triloop2.edgedict, name=triloop2.name)

    assert H.shape == (5, 4)
    duplicate_edge = ["ACD2"]
    newH = H.remove(duplicate_edge, level=0)
    assert newH.shape == (5, 3)

    assert H.shape == (5, 4)
    newH = H.remove(["E"], level=1)
    assert newH.shape == (4, 4)

    assert H.shape == (5, 4)
    newH = H.remove(["ACD"], level=0)
    assert newH.shape == (5, 3)

    # remove incidence in which the node is associated with other edges
    assert H.shape == (5, 4)
    newH = H.remove([("ACD", "E")])
    assert newH.shape == (5, 4)

    # edge case:
    # level defaults to 2, which handles the case of incidence pairs
    # the list of incidence pairs must be a list of tuples
    # if no tuples are given, the default behavior is to treat the list as a list of edges to be removed
    # if one of the edges in the list doesn't exist, it is ignored

    # case 1: level defaults to 2, list of uids is a list of edges and nodes
    assert H.shape == (5, 4)
    newH = H.remove(["ACD", "E"])
    assert newH.shape == (5, 3)

    # case 2: level defaults to 2, list of uids is a list of edges
    # removing 2 edges that have the node B. Node B is not associated with any other edge
    # new hypergraph should have 4 nodes and 2 edges
    assert H.shape == (5, 4)
    newH = H.remove(["AB", "BC"])
    assert newH.shape == (4, 2)

    # case 3: level defaults to 2, list of uids is a list of nodes
    # no change
    assert H.shape == (5, 4)
    newH = H.remove(list(triloop2.nodes))
    assert newH.shape == (5, 4)


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
    h2 = h.remove([5], 0)
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
