import pytest
import numpy as np
import networkx as nx
from hypernetx import Hypergraph, Entity, EntitySet
from hypernetx import HyperNetXError


def test_hypergraph_from_iterable_of_sets(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edges)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert H.degree("A") == 3
    assert H.number_of_edges() == 6
    assert H.number_of_nodes() == 7


def test_hypergraph_from_dict(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edgedict)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert H.degree("A") == 3
    assert H.size("R") == 2
    assert H.order() == 7


def test_hypergraph_custom_attributes(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edges)
    assert isinstance(H.__str__(), str)
    assert isinstance(H.__repr__(), str)
    assert H.__contains__("A")
    assert H.__len__() == 7
    nodes = [key for key in H.__iter__()]
    assert sorted(nodes) == ["A", "C", "E", "K", "T1", "T2", "V"]
    assert sorted(H.__getitem__("C")) == ["A", "E", "K"]


def test_hypergraph_static(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edges, static=True)
    assert len(H.edges) == 6
    assert len(H.nodes) == 7
    assert H.get_id("E") == 3
    assert list(H.get_linegraph(s=1)) == [0, 1, 2, 3, 4, 5]
    # H.get_name
    # H.translate


def test_hypergraph_from_dataframe(lesmis):
    df = lesmis.hypergraph.dataframe()
    H = Hypergraph.from_dataframe(df)
    assert H.shape == (40, 8)
    assert H.size(3) == 8
    assert H.degree("JA") == 3


def test_hypergraph_from_numpy_array(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph.from_numpy_array(sbs.arr)
    assert len(H.nodes) == 6
    assert len(H.edges) == 7
    assert H.dim("e5") == 2
    assert set(H.neighbors("v2")) == {"v0", "v5"}


def test_hypergraph_from_bipartite(sbsd_hypergraph):
    H = sbsd_hypergraph
    HB = Hypergraph.from_bipartite(H.bipartite())
    assert len(HB.edges) == 7
    assert len(HB.nodes) == 8
    assert HB.s_degree("T1") == 1


def test_hypergraph_from_entity_set(seven_by_six):
    sbs = seven_by_six
    entityset = EntitySet("_", sbs.edgedict)
    H = Hypergraph(entityset)
    assert H.edges.incidence_dict == sbs.edgedict
    assert H.s_degree("A") == 3
    assert H.dim("O") == 1
    assert len(H.edge_size_dist()) == 6
    assert len(H.edge_neighbors("S")) == 4


def test_add_node_to_edge(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edgedict)
    assert H.shape == (7, 6)
    # add node not already in hypergraph to edge
    # alreadyin hypergraph
    node = Entity("B")
    edge = H.edges["P"]
    H.add_node_to_edge(node, edge)
    assert H.shape == (8, 6)
    # add edge with nodes already in hypergraph
    H.add_edge(Entity("Z", ["A", "B"]))
    assert H.shape == (8, 7)
    # add edge not in hypergraph with nodes not in hypergraph
    H.add_edge(Entity("Y", ["M", "N"]))
    assert H.shape == (10, 8)


def test_remove_edge(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edgedict)
    assert H.shape == (7, 6)
    # remove an edge without removing any nodes
    H.remove_edge("P")
    assert H.shape == (7, 5)
    # remove an edge containing a singleton ear
    H.remove_edge("O")
    assert H.shape == (6, 4)


def test_remove_node():
    a, b, c, d = "a", "b", "c", "d"
    hbug = Hypergraph({0: [a, b], 1: [a, c], 2: [a, d]})
    assert a in hbug.nodes
    assert a in hbug.edges[0]
    assert a in hbug.edges[1]
    assert a in hbug.edges[2]
    hbug.remove_node(a)
    assert a not in hbug.nodes
    assert a not in hbug.edges[0]
    assert a not in hbug.edges[1]
    assert a not in hbug.edges[2]


def test_matrix(sbs_hypergraph):
    H = sbs_hypergraph
    assert H.incidence_matrix().todense().shape == (7, 6)
    assert H.adjacency_matrix(s=2).todense().shape == (7, 7)
    assert H.edge_adjacency_matrix().todense().shape == (6, 6)
    assert H.auxiliary_matrix().todense().shape == (6, 6)


def test_collapse_edges(sbsd_hypergraph):
    H = sbsd_hypergraph
    assert len(H.edges) == 7
    HC = H.collapse_edges()
    assert len(HC.edges) == 6


def test_collapse_nodes(sbsd_hypergraph):
    H = sbsd_hypergraph
    assert len(H.nodes) == 8
    HC = H.collapse_nodes()
    assert len(HC.nodes) == 7


def test_collapse_nodes_and_edges(sbsd_hypergraph):
    H = sbsd_hypergraph
    HC2 = H.collapse_nodes_and_edges()
    assert len(H.edges) == 7
    assert len(HC2.edges) == 6
    assert len(H.nodes) == 8
    assert len(HC2.nodes) == 7


def test_restrict_to_edges(sbs_hypergraph):
    H = sbs_hypergraph
    HS = H.restrict_to_edges(["P", "O"])
    assert len(H.edges) == 6
    assert len(HS.edges) == 2


def test_restrict_to_nodes(sbs_hypergraph):
    H = sbs_hypergraph
    assert len(H.nodes) == 7
    H1 = H.restrict_to_nodes(["A", "E", "K"])
    assert len(H.nodes) == 7
    assert len(H1.nodes) == 3
    assert len(H1.edges) == 5
    assert "C" in H.edges["P"]
    assert not "C" in H1.edges["P"]


def test_remove_from_restriction(triloop):
    h = triloop.hypergraph
    h1 = h.restrict_to_nodes(h.neighbors("A")).remove_node("A")
    assert "A" not in h1
    assert "A" not in h1.edges["ACD"]


def test_toplexes(sbsd_hypergraph):
    H = sbsd_hypergraph
    T = H.toplexes()
    assert len(T.nodes) == 8
    assert len(T.edges) == 5
    T = T.collapse_nodes()
    assert len(T.nodes) == 7


def test_is_connected():
    setsystem = [{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7}, {5, 6, 8}]
    h = Hypergraph(setsystem)
    assert h.is_connected() == True
    assert h.is_connected(s=2) == False
    assert h.is_connected(s=2, edges=True) == True
    assert h.is_connected(s=3, edges=True) == False


def test_singletons():
    E = {1: {2, 3, 4, 5}, 6: {2, 5, 7, 8, 9}, 10: {11}, 12: {13}, 14: {7}}
    h = Hypergraph(E)
    assert h.shape == (9, 5)
    singles = h.singletons()
    assert len(singles) == 2
    h.remove_edges(singles)
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
    assert list(h.connected_components(edges=True)) == [{"0", "1", "2", "3"}]
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
    assert list(h.s_connected_components()) == [{"0", "1", "2", "3"}]
    assert list(h.s_connected_components(s=2)) == [{"1", "2", "3"}]
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

def test_size(seven_by_six):
    sbs = seven_by_six
    h = Hypergraph(sbs.edgedict)
    assert h.size('S') == 4
    assert h.size('S',{'T2','V'}) == 2
    assert h.size('S',{'T1','T2'}) == 1
    assert h.size('S',{'T2'}) == 1
    assert h.size('S',{'T1'}) == 0
    assert h.size('S',{}) == 0

def test_diameter(seven_by_six):
    sbs = seven_by_six
    h = Hypergraph(sbs.edgedict)
    assert h.diameter() == 3
    with pytest.raises(Exception) as excinfo:
        h.diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_node_diameters(seven_by_six):
    sbs = seven_by_six
    h = Hypergraph(sbs.edgedict)
    assert h.node_diameters()[0] == 3
    assert h.node_diameters()[2] == [{"A", "C", "E", "K", "T1", "T2", "V"}]


def test_edge_diameter(seven_by_six):
    sbs = seven_by_six
    h = Hypergraph(sbs.edgedict)
    assert h.edge_diameter() == 3
    assert h.edge_diameters()[2] == [{"I", "L", "O", "P", "R", "S"}]
    with pytest.raises(Exception) as excinfo:
        h.edge_diameter(s=2)
    assert "Hypergraph is not s-connected." in str(excinfo.value)


def test_bipartite(sbs_hypergraph):
    from networkx.algorithms import bipartite

    h = sbs_hypergraph
    b = h.bipartite()
    assert bipartite.is_bipartite(b)


def test_dual(sbs_hypergraph):
    H = sbs_hypergraph
    HD = H.dual()
    assert set(H.nodes) == set(HD.edges)
    assert set(H.edges) == set(HD.nodes)


def test_distance(lesmis):
    h = lesmis.hypergraph
    assert h.distance("ME", "FN") == 2
    assert h.distance("ME", "FN", s=2) == 3
    assert h.distance("ME", "FN", s=3) == np.inf


def test_edge_distance(lesmis):
    h = lesmis.hypergraph
    assert h.edge_distance(1, 4) == 2
    h.remove_edge(5)
    assert h.edge_distance(1, 4) == 3
    assert h.edge_distance(1, 4, s=2) == np.inf


def test_dataframe(lesmis):
    h = lesmis.hypergraph
    df = h.dataframe()
    assert np.allclose(np.array(np.sum(df)), np.array([10, 9, 8, 4, 8, 3, 12, 6]))
