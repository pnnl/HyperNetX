from numpy.core.function_base import geomspace
from hypernetx import Hypergraph
from numpy.random import geometric
from scipy.special import binom
import random

def uniform_Erdos_Renyi(n, m, p, name=None):
    index = -1
    index += geometric(p)
    max_index = binom(n, m)
    edgelist = list()
    while index < max_index:
        edge = _get_edge_from_index(index, n, m)
        edgelist.append(edge)
        index += geometric(p)
    
    return Hypergraph(edgelist, name=name)

def uniform_configuration_model(k, name=None):
    edgelist = list()
    return Hypergraph(edgelist, name=name)

def uniform_Chung_Lu(k, name=None):
    edgelist = list()
    return Hypergraph(edgelist, name=name)

def uniform_SBM(group_memberships, m, p_tensor, name=None):
    n = len(group_memberships)
    p_max = max(p_tensor)
    index = -1
    index += geometric(p_max)
    max_index = binom(n, m)
    edgelist = list()
    while index < max_index:
        edge = _get_edge_from_index(index, n, m)
        # rejection sampling
        if random.random() <= p_tensor[tuple(group_memberships[edge])]/p_max:
            edgelist.append(edge)
        index += geometric(p_max)
    
    return Hypergraph(edgelist, name=name)

def uniform_DCSBM(k, group_memberships, m, p_tensor, name=None):
    n = len(group_memberships)
    p_max = max(p_tensor)
    index = -1
    index += geometric(p_max)
    max_index = binom(n, m)
    edgelist = list()
    while index < max_index:
        edge = _get_edge_from_index(index, n, m)
        # rejection sampling
        if random.random() <= p_tensor[group_memberships[edge]]/p_max:
            edgelist.append(edge)
        index += geometric(p_max)
    return Hypergraph(edgelist, name=name)

def _get_edge_from_index(index, n, m):
    return 0