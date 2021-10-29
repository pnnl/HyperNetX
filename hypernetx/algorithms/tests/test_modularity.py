import numpy as np
import pytest
import warnings
from hypernetx.algorithms.hypergraph_modularity import *
import random
import hypernetx as hnx

warnings.simplefilter("ignore")


def test_precompute(modularityexample):
    HG = modularityexample.HG
    HG = precompute_attributes(HG)
    assert HG.nodes['F'].strength == 3
    assert HG.total_weight == 6
    assert HG.edges['e2'].weight == 1


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
    assert {'A', 'B', 'C'} in kumar(HG)
    assert {'C', 'A', 'B'} in last_step(HG, A4)
