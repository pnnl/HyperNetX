import pytest
import os
import itertools as it
import networkx as nx
import pandas as pd
import numpy as np

from hypernetx import Hypergraph, HarryPotter, LesMis as LM
from hypernetx.classes.helpers import create_dataframe
from collections import OrderedDict, defaultdict, namedtuple


class SevenBySix:
    """Example hypergraph with 7 nodes and 6 edges."""

    def __init__(self):
        # Nodes
        SBS_Nodes = namedtuple("SBS_Nodes", "A C E K T1 T2 V")
        a, c, e, k, t1, t2, v = nd = ("A", "C", "E", "K", "T1", "T2", "V")
        self.nodes = SBS_Nodes(a, c, e, k, t1, t2, v)

        # Edges
        SBS_Edges = namedtuple("SBS_Edges", "I L O P R S")
        i, l, o, p, r, s = ("I", "L", "O", "P", "R", "S")
        self.edges = SBS_Edges(i, l, o, p, r, s)

        # Labels
        self.labels = OrderedDict(
            [
                ("edges", list(self.edges)),
                ("nodes", list(self.nodes)),
            ]
        )

        # define edges
        self.edges_list = [{a, c, k}, {a, e}, {a, k, t2, v}, {c, e}, {t1, t2}, {k, t2}]
        self.edgedict = OrderedDict(
            [
                (p, {a, c, k}),
                (r, {a, e}),
                (s, {a, k, t2, v}),
                (l, {c, e}),
                (o, {t1, t2}),
                (i, {k, t2}),
            ]
        )
        self.dataframe = create_dataframe(self.edgedict)

        # row = number of nodes = 6
        # columns = number of edges = 7
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

        self.data = np.array(
            [
                [3, 0],
                [3, 1],
                [3, 3],
                [4, 0],
                [4, 2],
                [5, 0],
                [5, 3],
                [5, 5],
                [5, 6],
                [1, 1],
                [1, 2],
                [2, 4],
                [2, 5],
                [0, 3],
                [0, 5],
            ]
        )


@pytest.fixture
def sbs() -> SevenBySix:
    return SevenBySix()


class TriLoop:
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D = "A", "B", "C", "D"
        AB, BC, ACD = "AB", "BC", "ACD"
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}}


@pytest.fixture
def triloop():
    return TriLoop()


class TriLoop2:
    """Triloop example with redundant node and edge"""

    def __init__(self):
        # Nodes
        A, B, C, D, E = "A", "B", "C", "D", "E"
        # Edges
        AB, BC, ACD, ACD2 = "AB", "BC", "ACD", "ACD2"
        self.nodes = set([A, B, C, D, E])

        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D, E}, ACD2: {A, C, D, E}}
        self.name = "TriLoop2"


@pytest.fixture
def triloop2():
    return TriLoop2()


class SBSDupes:
    def __init__(self):
        # Nodes
        a, c, e, f, k, t1, t2, v = nd = ("A", "C", "E", "F", "K", "T1", "T2", "V")
        self.nodes = (a, c, e, f, k, t1, t2, v)

        # Edges
        i, l, m, o, p, r, s = ("I", "L", "M", "O", "P", "R", "S")
        self.edges = (i, l, m, o, p, r, s)

        self.edgedict = OrderedDict(
            [
                (i, {k, t2}),
                (l, {c, e, f}),
                (m, {c, e, f}),
                ("O", {"T1", "T2"}),
                ("P", {"A", "C", "K"}),
                ("R", {"A", "E", "F"}),
                ("S", {"A", "K", "T2", "V"}),
            ]
        )

        self.dataframe = create_dataframe(self.edgedict)


@pytest.fixture
def sbs_dupes():
    return SBSDupes()


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


@pytest.fixture
def lesmis():
    return LesMis()


@pytest.fixture
def sample_df():
    fname = os.path.join(os.path.dirname(__file__), "sample.csv")
    return pd.read_csv(fname, index_col=0)


#### Old fixtures not in use


class CompleteBipartite:
    def __init__(self, n1, n2):
        self.g = nx.complete_bipartite_graph(n1, n2)
        self.left, self.right = nx.bipartite.sets(self.g)


@pytest.fixture
def sbs_graph():
    sbs = SevenBySix()
    edges = set()
    for _, e in sbs.edgedict.items():
        edges.update(it.combinations(e, 2))
    G = nx.Graph(name="sbsg")
    G.add_edges_from(edges)
    return G


@pytest.fixture
def G():
    return nx.karate_club_graph()


@pytest.fixture
def H():
    G = nx.karate_club_graph()
    return Hypergraph({f"e{i}": e for i, e in enumerate(G.edges())})


@pytest.fixture
def bipartite_example():
    from networkx.algorithms import bipartite

    return bipartite.random_graph(10, 5, 0.4, 0)


@pytest.fixture
def complete_bipartite_example():
    return CompleteBipartite(2, 3).g


@pytest.fixture
def dataframe_example():
    M = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]])
    index = ["A", "B", "C"]
    columns = ["a", "b", "c", "d"]
    return pd.DataFrame(M, index=index, columns=columns)


@pytest.fixture
def array_example():
    return np.array(
        [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    )
