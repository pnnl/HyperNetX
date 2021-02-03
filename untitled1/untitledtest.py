import hypernetx as hnx
import numpy as np


class TriLoop():
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D = 'A', 'B', 'C', 'D'
        AB, BC, ACD = 'AB', 'BC', 'ACD'
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}}
        self.hypergraph = hnx.Hypergraph(self.edgedict, name='TriLoop')


def test_homology_basis(triloop):
    Ck = {k: hnx.kchainbasis(triloop.hypergraph, k) for k in range(0, 3)}
    bd = {k: hnx.bkMatrix(Ck[k - 1], Ck[k]) for k in range(1, 3)}
    # assert np.array_equal(hnx.homology_basis(bd, 1), np.array([[1, 0, 1, 1, 1]]))
    # assert np.array_equal(hnx.homology_basis(bd, 1, C=Ck[1]), [[('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]])
    # assert np.array_equal(hnx.homology_basis(bd, 1, C=Ck[1], shortest=True)[0], [[('A', 'B'), ('A', 'C'), ('B', 'C')]]), hnx.homology_basis(bd, 1, C=Ck[1], shortest=True)
    return hnx.homology_basis(bd, 1, C=Ck[1], shortest=True)


if __name__ == '__main__':

    t = TriLoop()

    print(test_homology_basis(t))
