import warnings
import hypernetx as hnx
import untitled_StaticEntity as us
import untitled_static_Hypergraph as uh
from untitled_decorators import not_implemented_for
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd
from hypernetx.exception import HyperNetXError
from untitled_decorators import not_implemented_for
from scipy.sparse import issparse, coo_matrix, dok_matrix
from untitled_harrypotter import HarryPotter

harry = HarryPotter()
E = harry.entity
ES = harry.sparseentity
SE = us.StaticEntitySet(ES.arr, ES.labels, 0, 1)
h = uh.Hypergraph(setsystem={'arr': SE.arr, 'labels': SE.labels}, static=True)
im = h.incidence_matrix(sparse=True, index=False)
am, amdx = h.adjacency_matrix(s=100, index=True, sparse=True)

ctr = hnx.HNXCount(0)


def check(v, m):
    assert v, f'line {m}'
    ctr()
    print('*', end='')


check(len(E.elements['Gryffindor']) == 5, 18)

check(SE.arr.shape == (7, 11), 22)

check(len(h.edges.elements_by_level()) == 7, 28)

check('Half-blood' in h, 33)

check(im.shape == (11, 7), 35)

check('Muggle' in h.nodes, 41)

check(h.edge_adjacency_matrix(index=False, sparse=True).shape == (7, 7), 43)

check(am.shape == (11, 11), 46)

check(amdx[2] == 'Pure-blood', 48)

print(f'\n...all {ctr.value} tests passed...')
