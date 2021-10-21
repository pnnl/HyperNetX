"""
Hypergraph_Modularity
---------------------
Modularity and clustering for hypergraphs using HyperNetX.
Adapted from F. Théberge's GitHub repository: `Hypergraph Clustering <https://github.com/ftheberge/Hypergraph_Clustering>`_ 
See Tutorial 13 in the tutorials folder for library usage.

References
---------- 
.. [1] Kumar T., Vaidyanathan S., Ananthapadmanabhan H., Parthasarathy S. and Ravindran B. "A New Measure of Modularity in Hypergraphs: Theoretical Insights and Implications for Effective Clustering". In: Cherifi H., Gaito S., Mendes J., Moro E., Rocha L. (eds) Complex Networks and Their Applications VIII. COMPLEX NETWORKS 2019. Studies in Computational Intelligence, vol 881. Springer, Cham. https://doi.org/10.1007/978-3-030-36687-2_24
.. [2] Kamiński  B., Prałat  P. and Théberge  F. "Community Detection Algorithm Using Hypergraph Modularity". In: Benito R.M., Cherifi C., Cherifi H., Moro E., Rocha L.M., Sales-Pardo M. (eds) Complex Networks & Their Applications IX. COMPLEX NETWORKS 2020. Studies in Computational Intelligence, vol 943. Springer, Cham. https://doi.org/10.1007/978-3-030-65347-7_13
.. [3] Kamiński  B., Poulin V., Prałat  P., Szufel P. and Théberge  F. "Clustering via hypergraph modularity", Plos ONE 2019, https://doi.org/10.1371/journal.pone.0224307
"""

from collections import Counter
import numpy as np
from functools import reduce
import igraph as ig
import itertools
from scipy.special import comb

################################################################################

# we use 2 representations for partitions (0-based part ids):
# (1) dictionary or (2) list of sets


def dict2part(D):
    """
    Given a dictionary mapping the part for each vertex, return a partition as a list of sets; inverse function to part2dict

    Parameters
    ----------
    D : dict
        Dictionary keyed by vertices with values equal to integer
        index of the partition the vertex belongs to

    Returns
    -------
    : list
        List of sets; one set for each part in the partition
    """
    P = []
    k = list(D.keys())
    v = list(D.values())
    for x in range(max(D.values()) + 1):
        P.append(set([k[i] for i in range(len(k)) if v[i] == x]))
    return P


def part2dict(A):
    """
    Given a partition (list of sets), returns a dictionary mapping the part for each vertex; inverse function
    to dict2part

    Parameters
    ----------
    A : list of sets
        a partition of the vertices

    Returns
    -------
    : dict
      a dictionary with {vertex: partition index}
    """
    x = []
    for i in range(len(A)):
        x.extend([(a, i) for a in A[i]])
    return {k: v for k, v in x}

################################################################################

def precompute_attributes(HG):
    """
    Precompute some values on hypergraph HG for faster computing of hypergraph modularity. 
    If HG is unweighted, v.weight is set to 1 for each vertex v in HG. 
    The weighted degree for each vertex v is stored in v.strength.
    The total edge weigths for each edge cardinality is stored in HG.d_weights.
    Binomial coefficients to speed-up modularity computation are stored in HG.bin_coef.

    This needs to be run before calling either modularity() or last_step().

    Parameters
    ----------
    HG : Hypergraph

    """
    # 1. compute node strenghts (weighted degrees)
    for v in HG.nodes:
        HG.nodes[v].strength = 0
    for e in HG.edges:
        try:
            w = HG.edges[e].weight
        except:
            w = 1
            # add unit weight if none to simplify other functions
            HG.edges[e].weight = 1
        for v in list(HG.edges[e]):
            HG.nodes[v].strength += w
    # 2. compute d-weights
    ctr = Counter([len(HG.edges[e]) for e in HG.edges])
    for k in ctr.keys():
        ctr[k] = 0
    for e in HG.edges:
        ctr[len(HG.edges[e])] += HG.edges[e].weight
    HG.d_weights = ctr
    HG.total_weight = sum(ctr.values())
    # 3. compute binomial coeffcients (modularity speed-up)
    bin_coef = {}
    for n in HG.d_weights.keys():
        for k in np.arange(n // 2 + 1, n + 1):
            bin_coef[(n, k)] = comb(n, k, exact=True)
    HG.bin_coef = bin_coef

################################################################################


def linear(d, c):
    """
    Edge contribution [3]_ for $d$-edge with $c$ vertices in the majority class.
    If $c > d/2$, return $c/d$ else return 0.
    This is the default choice for modularity() and last_step() functions.

    Parameters
    ----------
    d : int
        Number of vertices in an edge
    c : int
        Number of vertices in the majority class

    Returns
    -------
    float
    """
    return c / d if c > d / 2 else 0

# majority


def majority(d, c):
    """    
    Edge contribution[3]_ for $d$-edge with $c$ vertices in the majority class.
    If $c>d/2$, return 1 else return 0.

    Parameters
    ----------
    d : int
        Number of nodes in an edge
    c : int
        Number of nodes in the majority class

    Returns
    -------
    bool
    """
    return 1 if c > d / 2 else 0

# strict


def strict(d, c):
    """
    Edge contribution [3]_ for $d$-edge with $c$ vertices in the majority class.
    If $c==d$, return 1 else return 0.

    Parameters
    ----------
    d : int
        Number of nodes in an edge
    c : int
        Number of nodes in the majority class

    Returns
    -------
    bool
    """
    return 1 if c == d else 0

#########################################


def _compute_partition_probas(HG, A):
    """
    Compute vol(A_i)/vol(V) for each part A_i in A (list of sets)

    Parameters
    ----------
    HG : Hypergraph
     A : list of sets

    Returns
    -------
    : list
        normalized distribution of strengths in partition elements
    """
    p = []
    for part in A:
        vol = 0
        for v in part:
            vol += HG.nodes[v].strength
        p.append(vol)
    s = sum(p)
    return [i / s for i in p]


def _degree_tax(HG, Pr, wdc):
    """
    Computes the expected fraction of edges falling in 
    the partition in a random graph as per [2]_

    Parameters
    ----------
    HG : Hypergraph

    Pr : list
        Probability distribution
    wdc : func
        weight function (ex: strict, majority, linear)

    Returns
    -------
    float

    """
    DT = 0
    for d in HG.d_weights.keys():
        tax = 0
        for c in np.arange(d // 2 + 1, d + 1):
            for p in Pr:
                tax += p**c * (1 - p)**(d - c) * HG.bin_coef[(d, c)] * wdc(d, c)
        tax *= HG.d_weights[d]
        DT += tax
    DT /= HG.total_weight
    return DT


def _edge_contribution(HG, A, wdc):
    """
    Edge contribution from hypergraph with respect
    to partion A.

    Parameters
    ----------
    HG : Hypergraph

    A : list of sets

    wdc : func
        weight function (ex: strict, majority, linear)

    Returns
    -------
    : float

    """
    EC = 0
    for e in HG.edges:
        d = HG.size(e)
        for part in A:
            if HG.size(e, part) > d / 2:
                EC += wdc(d, HG.size(e, part)) * HG.edges[e].weight
    EC /= HG.total_weight
    return EC

# HG: HNX hypergraph
# A: partition (list of sets)
# wcd: weight function (ex: strict, majority, linear)


def modularity(HG, A, wdc=linear):
    """
    Computes modularity of a hypergraph with respect to partition A.

    Parameters
    ----------
    HG : Hypergraph
        Description
    A : list of lists
        Partition of the nodes in HG
    wdc : func, optional
        weight function (ex: strict, majority, linear) 

    Returns
    -------
    : float

    """
    Pr = _compute_partition_probas(HG, A)
    return _edge_contribution(HG, A, wdc) - _degree_tax(HG, Pr, wdc)

################################################################################


def two_section(HG):
    """
    Creates a random walk 2-section igraph with transition weights defined by the
    weights of the hyperedges.

    Parameters
    ----------
    HG : Hypergraph

    Returns
    -------
    G : igraph.Graph
    """
    s = []
    for e in HG.edges:
        E = HG.edges[e]
        # random-walk 2-section (preserve nodes' weighted degrees)
        if len(E)>1:
            try:
                w = HG.edges[e].weight / (len(E) - 1)
            except:
                w = 1 / (len(E) - 1)
            s.extend([(k[0], k[1], w) for k in itertools.combinations(E, 2)])
    G = ig.Graph.TupleList(s, weights=True).simplify(combine_edges='sum')
    return G

################################################################################


def kumar(HG, delta=.01):
    """
    Compute a partition of the vertices as per Kumar's algorithm [1]_


    Parameters
    ----------
    HG : Hypergraph

    delta : float, optional
        convergence stopping criterion

    Returns
    -------
    dict

    """
    # weights will be modified -- store initial weights
    W = {e: HG.edges[e].weight for e in HG.edges}  # uses edge id for reference instead of int
    # build graph
    G = two_section(HG)
    # apply clustering
    CG = G.community_multilevel(weights='weight')
    CH = []
    for comm in CG.as_cover():
        CH.append(set([G.vs[x]['name'] for x in comm]))

    # LOOP
    diff = 1
    ctr = 0
    while diff > delta:
        # re-weight
        diff = 0
        for e in HG.edges:
            edge = HG.edges[e]
            reweight = sum([1 / (1 + HG.size(e, c)) for c in CH]) * (HG.size(e) + len(CH)) / HG.number_of_edges()
            diff = max(diff, 0.5 * abs(edge.weight - reweight))
            edge.weight = 0.5 * edge.weight + 0.5 * reweight
        # re-run louvain
        # build graph
        G = two_section(HG)
        # apply clustering
        CG = G.community_multilevel(weights='weight')
        CH = []
        for comm in CG.as_cover():
            CH.append(set([G.vs[x]['name'] for x in comm]))
        ctr += 1
        if ctr > 50:  # this process sometimes gets stuck -- set limit
            break
    G.vs['part'] = CG.membership
    for e in HG.edges:
        HG.edges[e].weight = W[e]
    return dict2part({v['name']: v['part'] for v in G.vs})

################################################################################


def _delta_ec(HG, P, v, a, b, wdc):
    """
    Computes change in edge contribution --
    partition P, node v going from P[a] to P[b]

    Parameters
    ----------
    HG : Hypergraph

    P : list of sets

    v : int or str
        node identifier
    a : int

    b : int

    wdc : func
        weight function (ex: strict, majority, linear)

    Returns
    -------
    TYPE
        Description
    """
    Pm = P[a] - {v}
    Pn = P[b].union({v})
    ec = 0
    for e in list(HG.nodes[v].memberships):
        d = HG.size(e)
        w = HG.edges[e].weight
        ec += w * (wdc(d, HG.size(e, Pm)) + wdc(d, HG.size(e, Pn))
                   - wdc(d, HG.size(e, P[a])) - wdc(d, HG.size(e, P[b])))
    return ec / HG.total_weight


def _bin_ppmf(d, c, p):
    """
    exp. part of binomial pmf

    Parameters
    ----------
    d : int

    c : int

    p : float


    Returns
    -------
    float

    """
    return p**c * (1 - p)**(d - c)


def _delta_dt(HG, P, v, a, b, wdc):
    """
    Compute change in degree tax --
    partition P (list), node v going from P[a] to P[b]

    Parameters
    ----------
    HG : Hypergraph

    P : list of sets

    v : int or str
         node identifier
    a : int

    b : int

    wdc : func
        weight function (ex: strict, majority, linear)

    Returns
    -------
    : float

    """
    s = HG.nodes[v].strength
    vol = sum([HG.nodes[v].strength for v in HG.nodes])
    vola = sum([HG.nodes[v].strength for v in P[a]])
    volb = sum([HG.nodes[v].strength for v in P[b]])
    volm = (vola - s) / vol
    voln = (volb + s) / vol
    vola /= vol
    volb /= vol
    DT = 0

    for d in HG.d_weights.keys():
        x = 0
        for c in np.arange(int(np.floor(d / 2)) + 1, d + 1):
            x += HG.bin_coef[(d, c)] * wdc(d, c) * (_bin_ppmf(d, c, voln) + _bin_ppmf(d, c, volm)
                                                    - _bin_ppmf(d, c, vola) - _bin_ppmf(d, c, volb))
        DT += x * HG.d_weights[d]
    return DT / HG.total_weight


def last_step(HG, L, wdc=linear, delta=.01):
    """
    Compute a partition of the vertices as per Last-Step algorithm.[2]_

    Simple H-based algorithm --
    try moving nodes between communities to optimize qH
    requires L: initial non-trivial partition

    Parameters
    ----------
    HG : Hypergraph

     L : list of sets

    wdc : func, optional
        weight function (ex: strict, majority, linear)
    delta : float, optional


    Returns
    -------
    : list of sets

    """
    A = L[:]  # we will modify this, copy
    D = part2dict(A)
    qH = 0
    while True:
        for v in list(np.random.permutation(list(HG.nodes))):
            c = D[v]
            s = list(set([c] + [D[i] for i in HG.neighbors(v)]))
            M = []
            if len(s) > 0:
                for i in s:
                    if c == i:
                        M.append(0)
                    else:
                        M.append(_delta_ec(HG, A, v, c, i, wdc) - _delta_dt(HG, A, v, c, i, wdc))
                i = s[np.argmax(M)]
                if c != i:
                    A[c] = A[c] - {v}
                    A[i] = A[i].union({v})
                    D[v] = i
        Pr = _compute_partition_probas(HG, A)
        q2 = _edge_contribution(HG, A, wdc) - _degree_tax(HG, Pr, wdc)
        if (q2 - qH) < delta:
            break
        qH = q2
    return [a for a in A if len(a) > 0]

################################################################################
