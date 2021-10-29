import pytest
import itertools as it
import networkx as nx
import hypernetx as hnx
import numpy as np


class TriLoop:
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D = "A", "B", "C", "D"
        AB, BC, ACD = "AB", "BC", "ACD"
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}}
        self.hypergraph = hnx.Hypergraph(self.edgedict, name="TriLoop")


class Fish:
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D, E, F, G, H = "A", "B", "C", "D", "E", "F", "G", "H"
        AB, BC, ACD, BEH, CF, AG = "AB", "BC", "ACD", "BEH", "CF", "AG"
        self.edgedict = {
            AB: {A, B},
            BC: {B, C},
            ACD: {A, C, D},
            BEH: {B, E, H},
            CF: {C, F},
            AG: {A, G},
        }
        self.hypergraph = hnx.Hypergraph(self.edgedict, name="Fish")
        thisstate = {
            "chains": {
                0: [("A",), ("B",), ("C",), ("D",), ("E",), ("F",), ("G",), ("H",)],
                1: [
                    ("A", "B"),
                    ("A", "C"),
                    ("A", "D"),
                    ("A", "G"),
                    ("B", "C"),
                    ("B", "E"),
                    ("B", "H"),
                    ("C", "D"),
                    ("C", "F"),
                    ("E", "H"),
                ],
                2: [("A", "C", "D"), ("B", "E", "H")],
                3: [],
            },
            "bkMatrix": {
                1: np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    ]
                ),
                2: np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [1, 0],
                        [0, 0],
                        [0, 0],
                        [0, 1],
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [0, 1],
                    ]
                ),
                3: np.array([[], []], dtype=np.int64),
            },
        }
        self.state = thisstate


class BigFish:
    """Example hypergraph with 4 2-cells forming a void and 1-2cell and 2 1-cells forming a loop"""

    def __init__(self):
        A, B, C, D, E, F, G, H, I = "A", "B", "C", "D", "E", "F", "G", "H", "I"
        AB, BC, ACD, BEH, CF, AG, ADI, ACI, CDI = (
            "AB",
            "BC",
            "ACD",
            "BEH",
            "CF",
            "AG",
            "ADI",
            "ACI",
            "CDI",
        )
        self.edgedict = {
            AB: {A, B},
            BC: {B, C},
            ACD: {A, C, D},
            BEH: {B, E, H},
            CF: {C, F},
            AG: {A, G},
            ADI: {A, D, I},
            ACI: {A, C, I},
            CDI: {C, D, I},
        }
        self.hypergraph = hnx.Hypergraph(self.edgedict, name="BigFish")


class SixByFive:
    """Example hypergraph with 6 nodes and 5 edges"""

    def __init__(self):
        mat = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 1],
                [0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ]
        ).transpose()
        self.hypergraph = hnx.Hypergraph.from_numpy_array(mat)


class ModularityExample:
    """
    ## build a hypergraph from a list of sets (the hyperedges)
    """

    def __init__(self):
        E = [{'A', 'B'}, {'A', 'C'}, {'A', 'B', 'C'}, {'A', 'D', 'E', 'F'}, {'D', 'F'}, {'E', 'F'}]
        self.E = E
        self.HG = hnx.Hypergraph(E, static=True)
        A1 = [{'A', 'B', 'C'}, {'D', 'E', 'F'}]
        A2 = [{'B', 'C'}, {'A', 'D', 'E', 'F'}]
        A3 = [{'A', 'B', 'C', 'D', 'E', 'F'}]
        A4 = [{'A'}, {'B'}, {'C'}, {'D'}, {'E'}, {'F'}]
        self.partitions = [A1, A2, A3, A4]


@pytest.fixture
def triloop():
    return TriLoop()


@pytest.fixture
def fish():
    return Fish()


@pytest.fixture
def bigfish():
    return BigFish()


@pytest.fixture
def sixbyfive():
    return SixByFive()


@pytest.fixture
def modularityexample():
    return ModularityExample()
