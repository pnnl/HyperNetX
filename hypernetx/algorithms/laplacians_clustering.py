# Copyright © 2021 Battelle Memorial Institute
# All rights reserved.

"""

Hypergraph Probability Transition Matrices, Laplacians, and Clustering
======================================================================
We contruct hypergraph random walks utilizing optional "edge-dependent vertex weights", which are 
weights associated with each vertex-hyperedge pair (i.e. cell weights on the incidence matrix).
The probability transition matrix of this random walk is used to construct a normalized Laplacian 
matrix for the hypergraph. That normalized Laplacian then serves as the input for a spectral clustering
algorithm. This spectral clustering algorithm, as well as the normalized Laplacian and other details of
this methodology are described in 

K. Hayashi, S. Aksoy, C. Park, H. Park, "Hypergraph random walks, Laplacians, and clustering", 
Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020.
https://doi.org/10.1145/3340531.3412034

Please direct any inquiries concerning the clustering module to Sinan Aksoy, sinan.aksoy@pnnl.gov

"""

import numpy as np
from collections import defaultdict
import networkx as nx
import warnings
import sys
from scipy.sparse import csr_matrix, coo_matrix, diags, find, identity
from scipy.sparse.linalg import eigs
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import preprocessing
from functools import partial
from hypernetx import HyperNetXError

try:
    import nwhy

    nwhy_available = True
except:
    nwhy_available = False

sys.setrecursionlimit(10000)

__all__ = [
    "prob_trans",
    "get_pi",
    "norm_lap",
    "spec_clus",
]


def prob_trans(H, weights=False, index=True, check_connected=True):
    """
    The probability transition matrix of a random walk on the vertices of a hypergraph.
    At each step in the walk, the next vertex is chosen by:

    1. Selecting a hyperedge e containing the vertex with probability proportional to w(e)
    2. Selecting a vertex v within e with probability proportional to a \gamma(v,e)

    If weights are not specified, then all weights are uniform and the walk is equivalent
    to a simple random walk.
    If weights are specified, the hyperedge weights w(e) are determined from the weights
    \gamma(v,e).


    Parameters
    ----------
    H : hnx.Hypergraph
        The hypergraph must be connected, meaning there is a path linking any two
        vertices
    weights : bool, optional, default : False
         Use the cell_weights associated with the hypergraph
         If False, uniform weights are utilized.
    index : bool, optional
        Whether to return matrix index to vertex label mapping

    Returns
    -------
     P : scipy.sparse.csr.csr_matrix
         Probability transition matrix of the random walk on the hypergraph
     index: dict
         mapping from row and column indices to corresponding vertex label
    """
    # hypergraph must be connected
    if check_connected:
        if not H.is_connected():
            raise HyperNetXError("hypergraph must be connected")

    # if no weighting function, each step in the random walk is chosen uniformly at random.
    if weights == False:
        R, index, _ = H.incidence_matrix(index=True)
    else:
        R, index, _ = H.incidence_matrix(index=True, weights=True)

    # transpose incidence matrix for notational convenience
    R = R.transpose()

    # generates hyperedge weight matrix, has same nonzero pattern as incidence matrix,
    # with values determined by the edge-dependent vertex weight standard deviation
    edgeScore = {
        i: np.std(R.getrow(i).data) + 1 for i in range(R.shape[0])
    }  # hyperedge weights
    vals = [edgeScore[i] for i in R.nonzero()[0]]
    W = csr_matrix(
        (vals, (R.nonzero()[1], R.nonzero()[0])), shape=(R.shape[1], R.shape[0])
    )

    # generate diagonal degree matrices used to normalize probability transition matrix
    [rowSums] = R.sum(axis=1).flatten().tolist()
    D_E = diags([1 / x for x in rowSums])

    [rowSums] = W.sum(axis=1).flatten().tolist()
    D_V = diags([1 / x for x in rowSums])

    # probability transition matrix P
    P = D_V * W * D_E * R

    if index == False:
        return P
    else:
        return P, index


def get_pi(P):
    """
    Returns the eigenvector corresponding to the largest eigenvalue (in magnitude),
    normalized so its entries sum to 1. Intended for the probability transition matrix
    of a random walk on a (connected) hypergraph, in which case the output can
    be interpreted as the stationary distribution.

    Parameters
    ----------
    P : csr matrix
        Probability transition matrix

    Returns
    -------
     pi : numpy.ndarray
         Stationary distribution of random walk defined by P
    """
    rho, pi = eigs(
        np.transpose(P), k=1, return_eigenvectors=True
    )  # dominant eigenvector
    pi = np.real(pi / np.sum(pi)).flatten()  # normalize as prob distribution
    return pi


def norm_lap(H, weights=False, index=True):
    """
    Normalized Laplacian matrix of the hypergraph. Symmetrizes the probability transition
    matrix of a hypergraph random walk using the stationary distribution, using the digraph
    Laplacian defined in:

    Chung, Fan. "Laplacians and the Cheeger inequality for directed graphs."
    Annals of Combinatorics 9.1 (2005): 1-19.

    and studied in the context of hypergraphs in:

    Hayashi, K., Aksoy, S. G., Park, C. H., & Park, H.
    Hypergraph random walks, laplacians, and clustering.
    In Proceedings of CIKM 2020, (2020): 495-504.

    Parameters
    ----------
    H : hnx.Hypergraph
        The hypergraph must be connected, meaning there is a path linking any two
        vertices
    weight : bool, optional, default : False
         Uses cell_weights, if False, uniform weights are utilized.
    index : bool, optional
        Whether to return matrix-index to vertex-label mapping

    Returns
    -------
     P : scipy.sparse.csr.csr_matrix
         Probability transition matrix of the random walk on the hypergraph
     index: dict
         mapping from row and column indices to corresponding vertex label
    """
    if weights == None:
        P, index = prob_trans(H)
    else:
        P, index = prob_trans(H, weights=weights)
    pi = get_pi(P)
    gamma = diags(np.power(pi, 1 / 2)) * P * diags(np.power(pi, -1 / 2))
    L = identity(gamma.shape[0]) - (1 / 2) * gamma + gamma.transpose()

    if index:
        return L, index
    else:
        return L


def spec_clus(H, k, existing_lap=None, weights=False):
    """
    Hypergraph spectral clustering of the vertex set into k disjoint clusters
    using the normalized hypergraph Laplacian. Equivalent to the "RDC-Spec"
    Algorithm 1 in:

    Hayashi, K., Aksoy, S. G., Park, C. H., & Park, H.
    Hypergraph random walks, laplacians, and clustering.
    In Proceedings of CIKM 2020, (2020): 495-504.


    Parameters
    ----------
    H : hnx.Hypergraph
        The hypergraph must be connected, meaning there is a path linking any two
        vertices
    k : int
        Number of clusters
    existing_lap: csr matrix, optional
        Whether to use an existing Laplacian; otherwise, normalized hypergraph Laplacian
        will be utilized
    weights : bool, optional
         Use the cell_weights of the hypergraph. If False uniform weights are used.

    Returns
    -------
     clusters : dict
         Vertex cluster dictionary, keyed by integers 0,...,k-1, with lists of
         vertices as values.
    """
    if existing_lap == None:
        if weights == None:
            L, index = norm_lap(H)
        else:
            L, index = norm_lap(H, weights=weights)
    else:
        L = existing_lap

    # compute top eigenvectors
    e, v = eigs(identity(L.shape[0]) - L, k=k, which="LM", return_eigenvectors=True)
    v = np.real(v)  # ignore zero complex parts
    v = preprocessing.normalize(v, norm="l2", axis=1)  # normalize
    U = np.array(v)
    km = KMeans(init="k-means++", n_clusters=k, random_state=0)  # k-means
    km.fit(U)
    d = km.labels_

    # organize cluster assingments in dictionary of form cluster #: ips
    clusters = {i: [] for i in range(k)}
    for i in range(len(index)):
        clusters[d[i]].append(index[i])

    return clusters
