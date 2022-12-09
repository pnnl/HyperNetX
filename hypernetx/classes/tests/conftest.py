import os

import pytest
import networkx as nx
import pandas as pd
import numpy as np

import hypernetx as hnx

from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class SevenBySix:
    nodes_t: tuple  # exactly size 7
    edges_t: tuple  # exactly size 6
    edges: list = field(init=False)
    nodes: set = field(init=False)
    edgedict: OrderedDict = field(init=False)
    arr: np.array = field(init=False)
    labels: OrderedDict = field(init=False)
    data: np.array = field(init=False)

    def __post_init__(self):
        self.edges = [
            {self.nodes_t[0], self.nodes_t[1], self.nodes_t[3]},
            {self.nodes_t[0], self.nodes_t[2]},
            {self.nodes_t[0], self.nodes_t[3], self.nodes_t[5], self.nodes_t[6]},
            {self.nodes_t[1], self.nodes_t[2]},
            {self.nodes_t[4], self.nodes_t[5]},
            {self.nodes_t[3], self.nodes_t[5]},
        ]
        self.nodes = set(self.nodes_t)
        self.edgedict = OrderedDict(
            [
                (self.edges_t[3], self.edges[0]),
                (self.edges_t[4], self.edges[1]),
                (self.edges_t[5], self.edges[2]),
                (self.edges_t[1], self.edges[3]),
                (self.edges_t[2], self.edges[4]),
                (self.edges_t[0], self.edges[5]),
            ]
        )
        self.arr = np.array(
            [
                [0, 0, 0, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 1, 1],
            ]
        )
        self.labels = OrderedDict(
            [
                (
                    "edges",
                    [
                        self.edges_t[3],
                        self.edges_t[4],
                        self.edges_t[5],
                        self.edges_t[1],
                        self.edges_t[2],
                        self.edges_t[0],
                    ],
                ),
                ("nodes", self.nodes_t),
            ]
        )
        self.data = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 2],
                [2, 4],
                [2, 5],
                [3, 1],
                [3, 3],
                [4, 5],
                [4, 6],
                [5, 0],
                [5, 5],
            ]
        )


class TriLoop:
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D = "A", "B", "C", "D"
        AB, BC, ACD = "AB", "BC", "ACD"
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}}
        self.hypergraph = hnx.Hypergraph(self.edgedict, name="TriLoop")


class SBSDupes:
    def __init__(self):
        self.edgedict = OrderedDict(
            [
                ("I", {"K", "T2"}),
                ("L", {"C", "E", "F"}),
                ("M", {"C", "E", "F"}),
                ("O", {"T1", "T2"}),
                ("P", {"A", "C", "K"}),
                ("R", {"A", "E", "F"}),
                ("S", {"A", "K", "T2", "V"}),
            ]
        )


class LesMis:
    def __init__(self):
        self.edgedict = OrderedDict(
            [
                (1, {"CL", "CV", "GE", "GG", "MB", "MC", "ME", "MY", "NP", "SN"}),
                (2, {"IS", "JL", "JV", "MB", "ME", "MR", "MT", "MY", "PG"}),
                (3, {"BL", "DA", "FA", "FN", "FT", "FV", "LI", "ZE"}),
                (4, {"CO", "FN", "TH", "TM"}),
                (5, {"BM", "FF", "FN", "JA", "JV", "MT", "MY", "VI"}),
                (6, {"FN", "JA", "JV"}),
                (
                    7,
                    {
                        "BM",
                        "BR",
                        "CC",
                        "CH",
                        "CN",
                        "FN",
                        "JU",
                        "JV",
                        "PO",
                        "SC",
                        "SP",
                        "SS",
                    },
                ),
                (8, {"FN", "JA", "JV", "PO", "SP", "SS"}),
            ]
        )
        self.hypergraph = hnx.Hypergraph(self.edgedict)


class Dataframe:
    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), "sample.csv")
        self.df = pd.read_csv(fname, index_col=0)


@pytest.fixture
def seven_by_six():
    return SevenBySix(
        ("A", "C", "E", "K", "T1", "T2", "V"), ("I", "L", "O", "P", "R", "S")
    )


@pytest.fixture
def ent7x6(seven_by_six):
    return hnx.Entity(data=np.asarray(seven_by_six.data), labels=seven_by_six.labels)


@pytest.fixture
def sbs_hypergraph(seven_by_six):
    return hnx.Hypergraph(seven_by_six.edgedict, name="sbsh")


@pytest.fixture
def triloop():
    return TriLoop()


@pytest.fixture
def sbsd_hypergraph():
    sbsd = SBSDupes()
    return hnx.Hypergraph(sbsd.edgedict)


@pytest.fixture
def lesmis():
    return LesMis()


@pytest.fixture
def G():
    return nx.karate_club_graph()


@pytest.fixture
def H():
    G = nx.karate_club_graph()
    return hnx.Hypergraph({f"e{i}": e for i, e in enumerate(G.edges())})


@pytest.fixture
def bipartite_example():
    from networkx.algorithms import bipartite

    return bipartite.random_graph(10, 5, 0.4, 0)


@pytest.fixture
def dataframe():
    return Dataframe()


@pytest.fixture
def harry_potter():
    return hnx.HarryPotter()


@pytest.fixture
def harry_potter_ent(harry_potter):
    return hnx.Entity(data=np.asarray(harry_potter.data), labels=harry_potter.labels)
