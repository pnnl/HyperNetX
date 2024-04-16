import numpy as np
import pytest
import warnings
from hypernetx.algorithms.homology_mod2 import *
import hypernetx as hnx

warnings.simplefilter("ignore")


def test_kchainbasis(triloop, fish):
    C = kchainbasis(triloop.hypergraph, 1)
    assert len(C) == 5
    assert ("A", "B") in C
    assert ("B", "D") not in C
    C = kchainbasis(triloop.hypergraph, 2)
    assert len(C) == 1
    assert ("A", "C", "D") in C
    fh = fish.hypergraph
    assert kchainbasis(fh, 1) == fish.state["chains"][1]
    assert kchainbasis(fh, 2) == fish.state["chains"][2]


def test_interpret(triloop):
    C = kchainbasis(triloop.hypergraph, 1)
    assert interpret(C, [(0, 1, 0, 1, 0)]) == [[("A", "C"), ("B", "C")]]


def test_bkMatrix(triloop, fish):
    Ck = {k: hnx.kchainbasis(triloop.hypergraph, k) for k in range(0, 2)}
    bd = bkMatrix(Ck[0], Ck[1])
    assert np.array_equal(bd[0], np.array([1, 1, 1, 0, 0]))
    fh = fish.hypergraph
    Ck = fish.state["chains"]
    assert np.sum(np.equal(bkMatrix(Ck[0], Ck[1]), fish.state["bkMatrix"][1])) == 80
    assert np.sum(np.equal(bkMatrix(Ck[1], Ck[2]), fish.state["bkMatrix"][2])) == 20


def test_smith_normal_form_mod2(triloop):
    Ck = {k: hnx.kchainbasis(triloop.hypergraph, k) for k in range(0, 2)}
    bd = bkMatrix(Ck[0], Ck[1])
    P1, Q1, S1, P1inv = smith_normal_form_mod2(bd)
    assert np.array_equal(P1[0], np.array([1, 0, 0, 0]))
    assert np.array_equal(Q1[:, 2], np.array([0, 1, 1, 0, 0]))
    assert np.all(S1 == matmulreduce([P1, bd, Q1]))
    r = len(P1)
    assert np.all(np.eye(r) == logical_matmul(P1, P1inv))
    assert np.all(S1 == matmulreduce([P1, bd, Q1]))


def test_reduced_row_echelon_form_mod2():
    m = np.array(
        [
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    r = 3
    L, S, Linv = reduced_row_echelon_form_mod2(m)
    assert np.array_equal(
        S,
        np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    )
    assert np.array_equal(L, np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]]))
    assert np.all(S == logical_matmul(L, m))
    assert np.all(np.eye(3) == logical_matmul(L, Linv))


def test_betti(fish):
    bk = fish.state["bkMatrix"]
    Ck = fish.state["chains"]
    assert betti(bk)[1] == 1
    assert betti(bk)[2] == 0


def test_homology_basis(triloop):
    Ck = {k: hnx.kchainbasis(triloop.hypergraph, k) for k in range(0, 3)}
    bd = {k: hnx.bkMatrix(Ck[k - 1], Ck[k]) for k in range(1, 3)}
    assert np.array_equal(homology_basis(bd, k=1)[1], np.array([[1, 0, 1, 1, 1]]))


def test_hypergraph_homology_basis(triloop, bigfish):
    assert np.array_equal(
        hypergraph_homology_basis(triloop.hypergraph, 1, interpreted=True)[1][1],
        [[("A", "B"), ("A", "D"), ("B", "C"), ("C", "D")]],
    )
    assert np.array_equal(
        hypergraph_homology_basis(
            triloop.hypergraph, 1, interpreted=True, shortest=True
        )[1][1],
        [
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "C"),
            ]
        ],
    )
    assert np.array_equal(
        hypergraph_homology_basis(triloop.hypergraph, 1, interpreted=False)[1],
        np.array([[1, 0, 1, 1, 1]]),
    )
    basis, ibasis = hypergraph_homology_basis(bigfish.hypergraph, shortest=True)
    assert len(ibasis[2][0]) == 4
    assert ("A", "D", "I") in ibasis[2][0]
    assert len(ibasis[1][0]) == 3
    assert ("B", "C") in ibasis[1][0]
