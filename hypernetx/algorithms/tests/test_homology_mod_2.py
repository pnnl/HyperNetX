import numpy as np
import pytest,warnings
from hypernetx.algorithms.homology_mod2 import *
warnings.simplefilter("ignore")

def test_kchainbasis(triloop):
	C = kchainbasis(triloop.hypergraph,1)
	assert len(C) == 5
	assert ('A','B') in C
	assert ('B','D') not in C
	C = kchainbasis(triloop.hypergraph,2)
	assert len(C) == 1
	assert ('A','C','D') in C

def test_interpret(triloop):
	C = kchainbasis(triloop.hypergraph,1)
	assert interpret(C,[(0,1,0,1,0)]) == [[('A', 'C'), ('B', 'C')]]

def test_bkMatrix(triloop):
	Ck = {k:hnx.kchainbasis(triloop.hypergraph,k) for k in range(0,2)}
	bd = bkMatrix(Ck[0],Ck[1])
	assert np.array_equal(bd[0],np.array([1, 1, 1, 0, 0]))

def test_smith_normal_form_mod2(triloop):
	Ck = {k:hnx.kchainbasis(triloop.hypergraph,k) for k in range(0,2)}
	bd = bkMatrix(Ck[0],Ck[1])
	P1,Q1,S1,P1inv = smith_normal_form_mod2(bd)
	assert np.array_equal(P1[0],np.array([1, 0, 0, 0]))
	assert np.array_equal(Q1[:,2],np.array([0, 1, 1, 0, 0]))
	assert(np.all(S1 == matmulreduce([P1,bd,Q1],mod=2)))
	r = len(P1)
	assert(np.all(np.eye(r) == modmult(P1,P1inv,mod=2)))
	assert(np.all(S1 == matmulreduce([P1,bd,Q1],mod=2)))

def test_reduced_row_echelon_form_mod2():
	m = np.array([[0,1,0,1,0,0,1,0,0,1],
	[0,0,0,0,0,0,0,0,0,1],
	[1,0,0,1,0,0,0,0,0,0]])
	r = 3
	L,S,Linv = reduced_row_echelon_form_mod2(m)
	assert np.array_equal(S,np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))
	assert np.array_equal(L,np.array([[0, 0, 1],
	[1, 1, 0],
	[0, 1, 0]]))
	assert(np.all(S == modmult(L,m,mod=2)))
	assert(np.all(np.eye(3) == modmult(L,Linv,mod=2)))

def test_homology_basis(triloop):
	Ck = {k:hnx.kchainbasis(triloop.hypergraph,k) for k in range(0,3)}
	bd = {k: hnx.bkMatrix(Ck[k-1],Ck[k]) for k in range(1,3)}
	assert np.array_equal(homology_basis(bd,1),np.array([[1, 0, 1, 1, 1]]))
	assert np.array_equal(homology_basis(bd,1,C=Ck[1]),[[('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]])
	assert np.array_equal(homology_basis(bd,1,C=Ck[1],shortest=True)[0],[[('A', 'B'), ('A', 'C'), ('B', 'C')]])


def test_hypergraph_homology_basis(triloop):
	assert np.array_equal(hypergraph_homology_basis(triloop.hypergraph,1),[[('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]])
