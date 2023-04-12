from hypernetx import Hypergraph, Entity, EntitySet


def test_static_hypergraph_constructor_setsystem(sbs):
    H = Hypergraph(sbs.edgedict, static=True)
    assert isinstance(H.edges, EntitySet)
    assert H.isstatic == True
    assert H.nwhy == False
    assert H.shape == (7, 6)


def test_static_hypergraph_constructor_entity(sbs):
    E = Entity(data=sbs.data, labels=sbs.labels)
    H = Hypergraph(E, static=True)
    assert H.isstatic
    assert "A" in H.edges.incidence_dict["P"]


def test_static_hypergraph_get_id(sbs):
    H = Hypergraph(Entity(data=sbs.data, labels=sbs.labels))
    assert H.get_id("V") == 6
    assert H.get_id("S", edges=True) == 2


def test_static_hypergraph_get_name(sbs):
    H = Hypergraph(Entity(data=sbs.data, labels=sbs.labels))
    assert H.get_name(1) == "C"
    assert H.get_name(1, edges=True) == "R"


def test_static_hypergraph_get_linegraph(lesmis):
    H = Hypergraph(lesmis.edgedict, static=True)
    assert H.shape == (40, 8)
    G = H.get_linegraph(edges=True, s=2)
    assert G.number_of_edges, G.number_of_nodes == (8, 8)


def test_static_hypergraph_s_connected_components(lesmis):
    H = Hypergraph(lesmis.edgedict, static=True)
    assert {7, 8} in list(H.s_connected_components(edges=True, s=4))