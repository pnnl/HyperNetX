import warnings
import pytest


from hypernetx.algorithms.hypergraph_modularity import *

warnings.simplefilter("ignore")


def test_precompute(modularityexample):
    HG = modularityexample.HG
    HG = precompute_attributes(HG)
    assert HG.nodes["F"].strength == 3
    assert HG.total_weight == 6
    assert HG.edges[2].weight == 1


def test_modularity(modularityexample):
    HG = modularityexample.HG
    A1, A2, A3, A4 = modularityexample.partitions
    HG = precompute_attributes(HG)
    assert np.abs(modularity(HG, A1) - 0.41444526) < 10e-5
    assert np.abs(modularity(HG, A1, strict) - 0.434906995) < 10e-5
    assert np.abs(modularity(HG, A1, majority) - 0.39379753) < 10e-5


def test_clustering(modularityexample):
    HG = modularityexample.HG
    A1, A2, A3, A4 = modularityexample.partitions
    HG = precompute_attributes(HG)
    assert {"A", "B", "C"} in kumar(HG)
    assert {"C", "A", "B"} in last_step(HG, A4)


def test_conductance(modularityexample):
    HG = modularityexample.HG
    A1, A2, A3, A4 = modularityexample.partitions
    HG = precompute_attributes(HG)
    assert conductance(HG, A1[0]) == 1 / 2
    assert conductance(HG, A1[1]) == 4 / 7
    assert conductance(HG, A2[0]) == 7 / 4
    assert conductance(HG, A2[1]) == 7 / 11
    with pytest.raises(Exception):
        conductance(HG, A3[0])
    assert conductance(HG, A4[0]) == 11 / 4
    assert conductance(HG, A4[1]) == 5 / 2
    assert conductance(HG, A4[2]) == 5 / 2
    assert conductance(HG, A4[3]) == 3
    assert conductance(HG, A4[4]) == 3
    assert conductance(HG, A4[5]) == 8 / 3
