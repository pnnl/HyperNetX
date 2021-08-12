import numpy as np
import pytest
import warnings
from hypernetx.algorithms.laplacians_clustering import *
import random
import hypernetx as hnx

warnings.simplefilter("ignore")


def test_prob_trans(fish):
    h = fish.hypergraph.convert_to_static()
    P, _ = prob_trans(h)
    assert P[0, 0] - (4 / 9) < 10e-5
    assert P[0, 2] - (1 / 6) < 10e-5
    assert np.sum(np.abs(np.sum(P, axis=1) - np.ones(len(P.todense())))) < 1e-6
    assert P.shape == (8, 8)


def test_norm_lap(fish):
    h = fish.hypergraph.convert_to_static()
    L, _ = norm_lap(h)
    assert L.shape == (8, 8)
