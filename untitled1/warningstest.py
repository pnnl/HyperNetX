from hypernetx import *

class Fish():
    """Example hypergraph with 2 two 1-cells and 1 2-cell forming a loop"""

    def __init__(self):
        A, B, C, D, E, F, G, H = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'
        AB, BC, ACD, BEH, CF, AG = 'AB', 'BC', 'ACD', 'BEH', 'CF', 'AG'
        self.edgedict = {AB: {A, B}, BC: {B, C}, ACD: {A, C, D}, BEH: {B, E, H}, CF: {C, F}, AG: {A, G}}
        self.hypergraph = hnx.Hypergraph(self.edgedict, name='Fish')
        
fish = Fish()
fh = fish.hypergraph

C = dict()
for k in range(0,4):
    C[k] = kchainbasis(fh,k)
    
bd = dict()
for k in range(1,4):
    bd[k] = bkMatrix(C[k-1],C[k])

    
print(betti_numbers(fh,k=0))