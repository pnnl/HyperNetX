import pytest
import os
import itertools as it
import networkx as nx
import hypernetx as hnx
import pandas as pd


class SevenBySix():
    """Example hypergraph with 7 nodes and 6 edges."""

    def __init__(self):
        a, c, e, k, t1, t2, v = nd = ('A', 'C', 'E', 'K', 'T1', 'T2', 'V')
        i, l, o, p, r, s = ('I', 'L', 'O', 'P', 'R', 'S')
        self.edges = [{a, c, k}, {a, e}, {a, k, t2, v}, {c, e}, {t1, t2}, {k, t2}]
        self.nodes = set(nd)
        self.edgedict = {p: {a, c, k}, r: {a, e}, s: {a, k, t2, v}, l: {c, e}, o: {t1, t2}, i: {k, t2}}


class TriLoop():
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D = 'A', 'B', 'C', 'D'
        AB, BC, ACD = 'AB', 'BC', 'ACD'
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}}
        self.hypergraph = hnx.Hypergraph(self.edgedict, name='TriLoop')


class SBSDupes():
    def __init__(self):
        self.edgedict = {'I': {'K', 'T2'},
                         'L': {'C', 'E', 'F'},
                         'M': {'C', 'E', 'F'},
                         'O': {'T1', 'T2'},
                         'P': {'A', 'C', 'K'},
                         'R': {'A', 'E', 'F'},
                         'S': {'A', 'K', 'T2', 'V'}}


class LesMis():
    def __init__(self):
        self.edgedict = {1: {'CL', 'CV', 'GE', 'GG', 'MB', 'MC', 'ME', 'MY', 'NP', 'SN'},
                         2: {'IS', 'JL', 'JV', 'MB', 'ME', 'MR', 'MT', 'MY', 'PG'},
                         3: {'BL', 'DA', 'FA', 'FN', 'FT', 'FV', 'LI', 'ZE'},
                         4: {'CO', 'FN', 'TH', 'TM'},
                         5: {'BM', 'FF', 'FN', 'JA', 'JV', 'MT', 'MY', 'VI'},
                         6: {'FN', 'JA', 'JV'},
                         7: {'BM', 'BR', 'CC', 'CH', 'CN', 'FN', 'JU', 'JV', 'PO', 'SC', 'SP', 'SS'},
                         8: {'FN', 'JA', 'JV', 'PO', 'SP', 'SS'}}
        self.hypergraph = hnx.Hypergraph(self.edgedict)


class Dataframe():
    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), 'sample.csv')
        self.df = pd.read_csv(fname, index_col=0)


@ pytest.fixture
def seven_by_six():
    return SevenBySix()


@pytest.fixture
def triloop():
    return TriLoop()


@ pytest.fixture
def sbs_hypergraph():
    sbs = SevenBySix()
    return hnx.Hypergraph(sbs.edgedict, name='sbsh')


@ pytest.fixture
def sbs_graph():
    sbs = SevenBySix()
    edges = set()
    for _, e in sbs.edgedict.items():
        edges.update(it.combinations(e, 2))
    G = nx.Graph(name='sbsg')
    G.add_edges_from(edges)
    return G


@ pytest.fixture
def sbsd_hypergraph():
    sbsd = SBSDupes()
    return hnx.Hypergraph(sbsd.edgedict)


@ pytest.fixture
def lesmis():
    return LesMis()


@ pytest.fixture
def G():
    return nx.karate_club_graph()


@ pytest.fixture
def H():
    G = nx.karate_club_graph()
    return hnx.Hypergraph({f'e{i}': e
                           for i, e in enumerate(G.edges())})


@ pytest.fixture
def bipartite_example():
    from networkx.algorithms import bipartite
    return bipartite.random_graph(10, 5, .4, 0)


@ pytest.fixture
def dataframe():
    return Dataframe()
