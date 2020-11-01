import pytest
import numpy as np
import networkx as nx
from hypernetx import Hypergraph, Entity, EntitySet, StaticEntity, StaticEntitySet
from hypernetx import HyperNetXError


def test_static_hypergraph_constructor(harry_potter):
    E = harry_potter.entityset
    h = Hypergraph(E)
    assert h.shape == (11, 7)
    hr = h.restrict_to_edges(['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff'])
    assert hr.shape == (8, 4)
    hr = h.restrict_to_nodes(['Pure-blood', 'Pure-blood or half-blood', 'Half-blood'])
    assert hr.shape == (3, 6)
    hd = h.remove_static()
    assert hd.isstatic == False
    assert isinstance(hd.edges, EntitySet)
