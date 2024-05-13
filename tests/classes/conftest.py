from typing import Mapping, Iterable

import pytest
import os
import networkx as nx
import pandas as pd
import numpy as np

from hypernetx import Hypergraph
from hypernetx.classes.factory import dict_factory_method
from collections import OrderedDict, namedtuple


class SevenBySix:
    """Example hypergraph with 7 nodes and 6 edges."""

    def __init__(self):
        # Nodes
        SBS_Nodes = namedtuple("SBS_Nodes", "A C E K T1 T2 V")
        a, c, e, k, t1, t2, v = ("A", "C", "E", "K", "T1", "T2", "V")
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
        np_data = [
            [p, a],
            [p, c],
            [p, k],
            [r, a],
            [r, e],
            [s, a],
            [s, k],
            [s, t2],
            [s, v],
            [l, c],
            [l, e],
            [o, t1],
            [o, t2],
            [i, k],
            [i, t2],
        ]
        self.ndarray = np.array(np_data)

        self.properties = dict_factory_method(self.edgedict, 2)

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
def sevenbysix() -> SevenBySix:
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


class SevenBySixDupes:
    def __init__(self):
        # Nodes
        a, c, e, f, k, t1, t2, v = ("A", "C", "E", "F", "K", "T1", "T2", "V")
        self.nodes = (a, c, e, f, k, t1, t2, v)

        # Edges
        i, l, m, o, p, r, s = ("I", "L", "M", "O", "P", "R", "S")
        self.edges = (i, l, m, o, p, r, s)

        self.edgedict = OrderedDict(
            [
                (i, {k, t2}),
                (l, {c, e, f}),
                (m, {c, e, f}),
                (o, {t1, t2}),
                (p, {a, c, k}),
                (r, {a, e, f}),
                (s, {a, k, t2, v}),
            ]
        )

        self.dataframe = create_dataframe(self.edgedict)


@pytest.fixture
def sevenbysix_dupes():
    return SevenBySixDupes()


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


@pytest.fixture
def nx_graph():
    return nx.karate_club_graph()


@pytest.fixture
def hnx_graph_from_nx_graph(nx_graph):
    return Hypergraph({f"e{i}": e for i, e in enumerate(nx_graph.edges())})


def create_dataframe(data: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    """Create a valid pandas Dataframe that can be used to create a Hypergraph"""

    # convert the dictionary, data, into a Series where the dictionary keys are the index and the values
    # are the values of the Series
    # In addition, for any values that are list-like, transform each element of a list-like value into its own row
    # In other words, `explode()`, will enumerate all the incidences from `data`
    data_t = pd.Series(data=data).explode()

    # create a Dataframe from `data_t` with two columns labeled 0 and 1
    # the indexes from `data_t`, which is the keys of the original dictionary `data`, will be assigned to column 0
    # the values from `data_t`, which is the values from the original dictionary `data`, will be assigned to column 1

    return pd.DataFrame(data={0: data_t.index.to_list(), 1: data_t.values})
