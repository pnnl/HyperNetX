import pytest
import numpy as np
import networkx as nx
from hypernetx import Hypergraph, Entity, EntitySet, StaticEntity, StaticEntitySet
from hypernetx import HyperNetXError

try:
    import nwhy

    nwhy_available = True
except:
    nwhy_available = False


def test_static_hypergraph_constructor_setsystem_nwhy(seven_by_six):
    sbs = seven_by_six
    edict = sbs.edgedict
    H = Hypergraph(edict, use_nwhy=True)
    assert isinstance(H.edges, StaticEntitySet)
    assert H.isstatic == True
    if nwhy_available:
        assert H.nwhy == True
        assert isinstance(H.g, nwhy.NWHypergraph)
    else:
        assert H.nwhy == False


if nwhy_available:

    def test_nwhy(seven_by_six):
        sbs = seven_by_six
        edict = sbs.edgedict
        H = Hypergraph(edict, use_nwhy=True)
        assert H.nwhy
