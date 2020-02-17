import pytest
import itertools as it
import networkx as nx
import hypernetx as hnx

class TriLoop():
	"""Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""
	def __init__(self):
		A,B,C,D = 'A','B','C','D'
		AB,BC,ACD = 'AB','BC','ACD'
		self.edgedict = {AB:{A,B},BC:{B,C},ACD:{A,C,D}}
		self.hypergraph = hnx.Hypergraph(self.edgedict,name='TriLoop')

@pytest.fixture
def triloop():
	return TriLoop()