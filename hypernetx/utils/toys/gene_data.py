from networkx import bipartite
import os

__all__ = ["GeneData"]


class GeneData:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fname = f"{current_dir}/disGene.txt"
        self.disease_gene_network = bipartite.read_edgelist(fname, delimiter=" ")
        self.genes, self.diseases = bipartite.sets(self.disease_gene_network)
