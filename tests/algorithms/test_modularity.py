import warnings
import pytest

from hypernetx.algorithms.hypergraph_modularity import conductance

warnings.simplefilter("ignore")


def test_conductance(modularityexample):
    HG = modularityexample.HG
    A1, A2, A3, A4 = modularityexample.partitions
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
