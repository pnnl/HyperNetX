import pytest
import numpy as np
import networkx as nx
from hypernetx import Hypergraph, Entity, EntitySet, StaticEntity, StaticEntitySet
from hypernetx import HyperNetXError


def test_static_hypergraph_constructor_setsystem(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(sbs.edgedict, static=True)
    assert isinstance(H.edges, StaticEntitySet)
    assert H.isstatic == True
    assert H.nwhy == False
    assert H.shape == (7, 6)


def test_static_hypergraph_constructor_entity(seven_by_six):
    sbs = seven_by_six
    E = Entity("sbs", sbs.edgedict)
    H = Hypergraph(E, static=True)
    assert H.isstatic
    assert "A" in H.edges.incidence_dict["P"]


def test_static_hypergraph_get_id(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(StaticEntity(arr=sbs.arr, labels=sbs.labels))
    assert H.get_id("V") == 6
    assert H.get_id("S", edges=True) == 2


def test_static_hypergraph_get_name(seven_by_six):
    sbs = seven_by_six
    H = Hypergraph(StaticEntity(arr=sbs.arr, labels=sbs.labels))
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
