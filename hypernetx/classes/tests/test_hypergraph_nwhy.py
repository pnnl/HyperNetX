from hypernetx import Hypergraph, EntitySet

try:
    import nwhy

    nwhy_available = True
except ImportError:
    nwhy_available = False


def test_static_hypergraph_constructor_setsystem_nwhy(seven_by_six):
    sbs = seven_by_six
    edict = sbs.edgedict
    H = Hypergraph(edict, use_nwhy=True)
    assert isinstance(H.edges, EntitySet)
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
