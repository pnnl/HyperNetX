import numpy as np
import pytest
import warnings
from hypernetx.algorithms.s_centrality_measures import *

warnings.simplefilter("ignore")


def test_s_betweenness_centrality(fish):
    h = fish.hypergraph
    bc = s_betweenness_centrality(h)
    assert bc["AB"] == 0.2
    assert bc["ACD"] == 0.2
    assert bc["CF"] == 0
    bcd = s_betweenness_centrality(h.dual(), normalized=False)
    assert bcd["C"] == 7.5
    assert bcd["B"] == 10
    assert bcd["F"] == 0


def test_s_harmonic_centrality(sixbyfive):
    h = sixbyfive.hypergraph
    shcc = s_harmonic_centrality(h, s=1, normalized=True)
    s1 = {"e4": 0.66667, "e1": 0.66667, "e0": 0.66667, "e2": 0.66667, "e3": 0.66667}
    for e in h.edges:
        assert shcc[e] - s1[e] < 10e-5
    shcc = s_harmonic_centrality(h, s=2, normalized=True)
    s2 = {"e4": 0.66667, "e1": 0.58333, "e0": 0.66667, "e2": 0.58333, "e3": 0.5}
    for e in h.edges:
        assert shcc[e] - s2[e] < 10e-5
    shcc = s_harmonic_centrality(h, s=3, normalized=True)
    s3 = {"e0": 0.25, "e3": 0.25, "e4": 0.33333, "e1": 0.0, "e2": 0.0}
    for e in h.edges:
        assert shcc[e] - s3[e] < 10e-5


def test_s_eccentricity(sixbyfive):
    h = sixbyfive.hypergraph
    shcc = s_eccentricity(h, s=1)
    s1 = {"e0": 1, "e1": 1, "e2": 1, "e3": 1, "e4": 1}
    for e in h.edges:
        assert shcc[e] == s1[e]
    shcc = s_eccentricity(h, s=2)
    s2 = {"e0": 1, "e1": 2, "e2": 2, "e3": 2, "e4": 1}
    for e in h.edges:
        assert shcc[e] == s2[e]
    shcc = s_eccentricity(h, s=3)
    s3 = {"e0": 2, "e1": 0, "e2": 0, "e3": 2, "e4": 1}
    for e in h.edges:
        assert shcc[e] == s3[e]
