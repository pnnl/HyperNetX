import numpy as np
import pytest
import warnings
from hypernetx.algorithms.laplacians_clustering import *

warnings.simplefilter("ignore")


def test_prob_trans(fish):
    h = fish.hypergraph
    P, index = prob_trans(h)
    assert P[1, 1] == 0.5
    assert P[0, 0] - (4 / 9) < 10e-5
    assert P[0, 2] - (1 / 6) < 10e-5
    assert P.shape == (8, 8)


def test_norm_lap(fish):
    h = fish.hypergraph
    L, index = norm_lap(h)
    assert L[0, 0] - (11 / 9) < 10e-5
    assert L[0, 1] == 0
    assert L[2, 0] - (1 / 12) < 10e-5
    assert L.shape == (8, 8)
