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
import pandas as pd
import itertools
from scipy.stats import binom

try:
    import igraph as ig
except ModuleNotFoundError as e:
    print(
        f" {e}. If you need to use {__name__}, please install additional packages by running the following command: pip install .['all']"
    )
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


def linear(d, c):
    """
    Hyperparameter for hypergraph modularity [2]_ for d-edge with c vertices in the majority class.
    This is the default choice for modularity() and last_step() functions.

    Parameters
    ----------
    d : int
        Number of vertices in an edge
    c : int
        Number of vertices in the majority class

    Returns
    -------
    : float
      c/d if c>d/2 else 0
    """
    return c / d if c > d / 2 else 0


# majority


def majority(d, c):
    """
    Hyperparameter for hypergraph modularity [2]_ for d-edge with c vertices in the majority class.
    This corresponds to the majority rule [3]_

    Parameters
    ----------
    d : int
        Number of vertices in an edge
    c : int
        Number of vertices in the majority class

    Returns
    -------
    : bool
      1 if c>d/2 else 0

    """
    return 1 if c > d / 2 else 0


# strict


def strict(d, c):
    """
    Hyperparameter for hypergraph modularity [2]_ for d-edge with c vertices in the majority class.
    This corresponds to the strict rule [3]_

    Parameters
    ----------
    d : int
        Number of vertices in an edge
    c : int
        Number of vertices in the majority class

    Returns
    -------
    : bool
      1 if c==d else 0
    """
    return 1 if c == d else 0


#########################################


# wcd: weight function (ex: strict, majority, linear)
def modularity(HG, A, wdc=linear):
    """
    Computes modularity of hypergraph HG with respect to partition A.

    Parameters
    ----------
    HG : Hypergraph
        The hypergraph with some precomputed attributes via: precompute_attributes(HG)
    A : list of sets
        Partition of the vertices in HG
    wdc : func, optional
        Hyperparameter for hypergraph modularity [2]_

    Note
    ----
    For 'wdc', any function of the format w(d,c) that returns 0 when c <= d/2 and value in [0,1] otherwise can be used.
    Default is 'linear'; other supplied choices are 'majority' and 'strict'.

    Returns
    -------
    : float
      The modularity function for partition A on HG
    """
    ## all same edge weights?
    uniq = len(Counter(HG.edges.properties["weight"])) == 1

    ## Edge Contribution
    HG_id = HG.incidence_dict
    d = part2dict(A)
    L = [[d[i] for i in HG_id[x]] for x in HG_id]

    ## all same weight
    if uniq:
        _ctr = Counter([(Counter(l).most_common(1)[0][1], len(l)) for l in L])
        EC = sum([wdc(k[1], k[0]) * _ctr[k] for k in _ctr.keys() if k[0] > k[1] / 2])
    else:
        _keys = [(Counter(l).most_common(1)[0][1], len(l)) for l in L]
        _vals = list(HG.edge_props["weight"])  ## Thanks Brenda!!
        _df = pd.DataFrame(zip(_keys, _vals), columns=["key", "val"])
        _df = _df.groupby(by="key").sum()
        EC = sum(
            [wdc(k[1], k[0]) * v[0] for (k, v) in _df.iterrows() if k[0] > k[1] / 2]
        )

    ## Degree Tax
    if uniq:
        VolA = [sum([HG.degree(i) for i in k]) for k in A]
        Ctr = Counter([HG.size(i) for i in HG.edges])

    else:
        ## this is the bottleneck
        VolA = np.repeat(0, 1 + np.max(list(d.values())))
        m = np.max([HG.size(i) for i in HG.edges])
        Ctr = np.repeat(0, 1 + m)
        S = 0
        for e in HG.edges:
            w = HG.edges[e].weight
            Ctr[HG.size(e)] += w
            S += w
            for v in HG.edges[e]:
                VolA[d[v]] += w

    VolV = np.sum(VolA)
    VolA = [x / VolV for x in VolA]
    DT = 0

    if uniq:
        for d in Ctr.keys():
            Cnt = Ctr[d]
            for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                for Vol in VolA:
                    DT += Cnt * wdc(d, c) * binom.pmf(c, d, Vol)
        return (EC - DT) / HG.number_of_edges()
    else:
        for d in range(len(Ctr)):
            Cnt = Ctr[d]
            for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                for Vol in VolA:
                    DT += Cnt * wdc(d, c) * binom.pmf(c, d, Vol)
        return (EC - DT) / S


def conductance(H, A):
    """
    Computes conductance [4] of hypergraph HG with respect to partition A.

    Parameters
    ----------
    H : Hypergraph
        The hypergraph
    A : set
        Partition of the vertices in H

    Returns
    -------
    : float
      The conductance function for partition A on H
    """
    subset2 = [n for n in H.nodes if n not in A]
    if len(subset2) == 0:
        raise Exception("True subset is not allowed")
    ws = sum((H.degree(node) for node in A))
    was = 0
    for edge in H.edges:
        he_vertices = H.edges[edge]
        if len([n for n in he_vertices if n in A]) == 0:
            continue
        if len([n for n in he_vertices if n in subset2]) == 0:
            continue
        was += len(he_vertices)
    return was / ws


################################################################################


def two_section(HG):
    """
    Creates a random walk based [1]_ 2-section igraph Graph with transition weights defined by the
    weights of the hyperedges.

    Parameters
    ----------
    HG : Hypergraph

    Returns
    -------
     : igraph.Graph
       The 2-section graph built from HG
    """
    s = []
    for e in HG.edges:
        E = HG.edges[e]
        # random-walk 2-section (preserve nodes' weighted degrees)
        if len(E) > 1:
            try:
                w = HG.edges[e].weight / (len(E) - 1)
            except:
                w = 1 / (len(E) - 1)
            s.extend([(k[0], k[1], w) for k in itertools.combinations(E, 2)])
    G = ig.Graph.TupleList(s, weights=True).simplify(combine_edges="sum")
    return G


################################################################################


def kumar(HG, delta=0.01, verbose=False):
    """
    Compute a partition of the vertices in hypergraph HG as per Kumar's algorithm [1]_

    Parameters
    ----------
    HG : Hypergraph

    delta : float, optional
        convergence stopping criterion

    Returns
    -------
    : list of sets
       A partition of the vertices in HG

    """
    # weights will be modified -- store initial weights
    W = {
        e: HG.edges[e].weight for e in HG.edges
    }  # uses edge id for reference instead of int
    # build graph
    G = two_section(HG)
    # apply clustering
    CG = G.community_multilevel(weights="weight")
    CH = []
    for comm in CG.as_cover():
        CH.append(set([G.vs[x]["name"] for x in comm]))

    # LOOP
    diff = 1
    ctr = 0
    while diff > delta:
        # re-weight
        diff = 0
        for e in HG.edges:
            edge = HG.edges[e]
            reweight = (
                sum([1 / (1 + HG.size(e, c)) for c in CH])
                * (HG.size(e) + len(CH))
                / HG.number_of_edges()
            )
            diff = max(diff, 0.5 * abs(edge.weight - reweight))
            edge.weight = 0.5 * edge.weight + 0.5 * reweight
        if verbose:
            print("pass completed, max edge weight difference:", diff)

        # re-run louvain
        # build graph
        G = two_section(HG)
        # apply clustering
        CG = G.community_multilevel(weights="weight")
        CH = []
        for comm in CG.as_cover():
            CH.append(set([G.vs[x]["name"] for x in comm]))
        ctr += 1
        if ctr > 50:  # this process sometimes gets stuck -- set limit
            break
    G.vs["part"] = CG.membership
    for e in HG.edges:
        HG.edges[e].weight = W[e]
    return dict2part({v["name"]: v["part"] for v in G.vs})


################################################################################


## THIS ASSUMES WEIGHTED H
def _last_step_weighted(H, A, wdc, delta=0.01, verbose=False):
    qH = modularity(H, A, wdc=wdc)
    if verbose:
        print("initial qH:", qH)
    d = part2dict(A)

    ## initialize
    ## this is the bottleneck
    VolA = np.repeat(0, 1 + np.max(list(d.values())))
    m = np.max([H.size(i) for i in H.edges])
    ctr_sizes = np.repeat(0, 1 + m)
    S = 0
    for e in H.edges:
        w = H.edges[e].weight
        ctr_sizes[H.size(e)] += w
        S += w
        for v in H.edges[e]:
            VolA[d[v]] += w
    VolV = np.sum(VolA)
    dct_A = part2dict(A)

    ## loop
    while True:
        n_moves = 0
        for v in list(np.random.permutation(list(H.nodes))):
            dct_A_v = dct_A[v]
            H_id = [H.incidence_dict[x] for x in H.nodes[v].memberships]
            L = [[dct_A[i] for i in x] for x in H_id]

            ## ec portion before move
            _keys = [(Counter(l).most_common(1)[0][1], len(l)) for l in L]
            _vals = [H.edge_props["weight"][x] for x in H.nodes[v].memberships]
            _df = pd.DataFrame(zip(_keys, _vals), columns=["key", "val"])
            _df = _df.groupby(by="key").sum()
            ec = sum(
                [
                    wdc(k[1], k[0]) * val[0]
                    for (k, val) in _df.iterrows()
                    if k[0] > k[1] / 2
                ]
            )
            str_v = np.sum(_vals)  ## weighted degree

            ## DT portion before move
            dt = 0
            for d in range(len(ctr_sizes)):
                Cnt = ctr_sizes[d]
                for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                    dt += Cnt * wdc(d, c) * binom.pmf(c, d, VolA[dct_A_v] / VolV)

            ## move it?
            best = dct_A_v
            best_del_q = 0
            best_dt = 0
            for m in set([i for x in L for i in x]) - {dct_A_v}:
                dct_A[v] = m
                L = [[dct_A[i] for i in x] for x in H_id]
                ## EC
                _keys = [(Counter(l).most_common(1)[0][1], len(l)) for l in L]
                _vals = [H.edge_props["weight"][x] for x in H.nodes[v].memberships]
                _df = pd.DataFrame(zip(_keys, _vals), columns=["key", "val"])
                _df = _df.groupby(by="key").sum()
                ecp = sum(
                    [
                        wdc(k[1], k[0]) * val[0]
                        for (k, val) in _df.iterrows()
                        if k[0] > k[1] / 2
                    ]
                )

                ## DT
                del_dt = -dt
                for d in range(len(ctr_sizes)):
                    Cnt = ctr_sizes[d]
                    for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                        del_dt -= Cnt * wdc(d, c) * binom.pmf(c, d, VolA[m] / VolV)
                        del_dt += (
                            Cnt * wdc(d, c) * binom.pmf(c, d, (VolA[m] + str_v) / VolV)
                        )
                        del_dt += (
                            Cnt
                            * wdc(d, c)
                            * binom.pmf(c, d, (VolA[dct_A_v] - str_v) / VolV)
                        )
                del_q = ecp - ec - del_dt
                if del_q > best_del_q:
                    best_del_q = del_q
                    best = m
                    best_dt = del_dt

            if best_del_q > 0.1:  ## this avoids some precision issues
                n_moves += 1
                dct_A[v] = best
                VolA[m] += str_v
                VolA[dct_A_v] -= str_v
                VolV = np.sum(VolA)
            else:
                dct_A[v] = dct_A_v

        new_qH = modularity(H, dict2part(dct_A), wdc=wdc)
        if verbose:
            print(n_moves, "moves, new qH:", new_qH)
        if (new_qH - qH) < delta:
            break
        else:
            qH = new_qH
    return dict2part(dct_A)


## THIS ASSUMES UNWEIGHTED H
def _last_step_unweighted(H, A, wdc, delta=0.01, verbose=False):
    qH = modularity(H, A, wdc=wdc)
    if verbose:
        print("initial qH:", qH)

    ## initialize
    ctr_sizes = Counter([H.size(i) for i in H.edges])
    VolA = [sum([H.degree(i) for i in k]) for k in A]
    VolV = np.sum(VolA)
    dct_A = part2dict(A)

    while True:
        n_moves = 0
        for v in list(np.random.permutation(list(H.nodes))):
            dct_A_v = dct_A[v]
            H_id = [H.incidence_dict[x] for x in H.nodes[v].memberships]
            L = [[dct_A[i] for i in x] for x in H_id]
            deg_v = H.degree(v)

            ## assume unweighted - EC portion before
            _ctr = Counter([(Counter(l).most_common(1)[0][1], len(l)) for l in L])
            ec = sum(
                [wdc(k[1], k[0]) * _ctr[k] for k in _ctr.keys() if k[0] > k[1] / 2]
            )

            ## DT portion before
            dt = 0
            for d in ctr_sizes.keys():
                Cnt = ctr_sizes[d]
                for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                    dt += Cnt * wdc(d, c) * binom.pmf(c, d, VolA[dct_A_v] / VolV)

            ## move it?
            best = dct_A_v
            best_del_q = 0
            best_dt = 0
            for m in set([i for x in L for i in x]) - {dct_A_v}:
                dct_A[v] = m
                L = [[dct_A[i] for i in x] for x in H_id]
                ## assume unweighted - EC
                _ctr = Counter([(Counter(l).most_common(1)[0][1], len(l)) for l in L])
                ecp = sum(
                    [wdc(k[1], k[0]) * _ctr[k] for k in _ctr.keys() if k[0] > k[1] / 2]
                )
                ## DT
                del_dt = -dt
                for d in ctr_sizes.keys():
                    Cnt = ctr_sizes[d]
                    for c in np.arange(int(np.floor(d / 2 + 1)), d + 1):
                        del_dt -= Cnt * wdc(d, c) * binom.pmf(c, d, VolA[m] / VolV)
                        del_dt += (
                            Cnt * wdc(d, c) * binom.pmf(c, d, (VolA[m] + deg_v) / VolV)
                        )
                        del_dt += (
                            Cnt
                            * wdc(d, c)
                            * binom.pmf(c, d, (VolA[dct_A_v] - deg_v) / VolV)
                        )
                del_q = ecp - ec - del_dt
                if del_q > best_del_q:
                    best_del_q = del_q
                    best = m
                    best_dt = del_dt
            if best_del_q > 0.1:  ## this avoids some numerical precision issues
                n_moves += 1
                dct_A[v] = best
                VolA[m] += deg_v
                VolA[dct_A_v] -= deg_v
                VolV = np.sum(VolA)
            else:
                dct_A[v] = dct_A_v
        new_qH = modularity(H, dict2part(dct_A), wdc=wdc)
        if verbose:
            print(n_moves, "moves, new qH:", new_qH)
        if (new_qH - qH) < delta:
            break
        else:
            qH = new_qH
    return dict2part(dct_A)


def last_step(HG, A, wdc=linear, delta=0.01, verbose=False):
    """
    Given some initial partition L, compute a new partition of the vertices in HG as per Last-Step algorithm [2]_

    Note
    ----
    This is a very simple algorithm that tries moving nodes between communities to improve hypergraph modularity.
    It requires an initial non-trivial partition which can be obtained for example via graph clustering on the 2-section of HG,
    or via Kumar's algorithm.

    Parameters
    ----------
    HG : Hypergraph

    L : list of sets
      some initial partition of the vertices in HG

    wdc : func, optional
        Hyperparameter for hypergraph modularity [2]_

    delta : float, optional
            convergence stopping criterion
    verbose: boolean, optional
        If set, also returns progress after each pass through the vertices

    Returns
    -------
    : list of sets
      A new partition for the vertices in HG
    """
    ## all same edge weights?
    uniq = len(Counter(HG.edges.properties["weight"])) == 1

    if uniq:
        nls = _last_step_unweighted(HG, A, wdc=wdc, delta=delta, verbose=verbose)
    else:
        nls = _last_step_weighted(HG, A, wdc=wdc, delta=delta, verbose=verbose)
    return nls


################################################################################
