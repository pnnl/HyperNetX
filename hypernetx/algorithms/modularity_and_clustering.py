from collections import Counter
import numpy as np
from functools import reduce
import igraph as ig
import itertools

################################################################################

# we use 2 representations for partitions (0-based part ids):
# (1) dictionary or (2) list of sets


def dict2part(D):
    P = []
    k = list(D.keys())
    v = list(D.values())
    for x in range(max(D.values()) + 1):
        P.append(set([k[i] for i in range(len(k)) if v[i] == x]))
    return P


def part2dict(A):
    x = []
    for i in range(len(A)):
        x.extend([(a, i) for a in A[i]])
    return {k: v for k, v in x}

################################################################################


def factorial(n):
    if n < 2:
        return 1
    return reduce(lambda x, y: x * y, range(2, int(n) + 1))

# Precompute soe values on HNX hypergraph for computing qH faster


def HNX_precompute(HG):
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
            bin_coef[(n, k)] = factorial(n) / (factorial(k) * factorial(n - k))
    HG.bin_coef = bin_coef

################################################################################

# some weight function 'wdc' for d-edges with c-majority

# default: linear w.r.t. c


def linear(d, c):
    return c / d if c > d / 2 else 0

# majority


def majority(d, c):
    return 1 if c > d / 2 else 0

# strict


def strict(d, c):
    return 1 if c == d else 0

#########################################

# compute vol(A_i)/vol(V) for each part A_i in A (list of sets)


def compute_partition_probas(HG, A):
    p = []
    for part in A:
        vol = 0
        for v in part:
            vol += HG.nodes[v].strength
        p.append(vol)
    s = sum(p)
    return [i / s for i in p]

# degree tax


def DegreeTax(HG, Pr, wdc):
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


# edge contribution, A is list of sets
def EdgeContribution(HG, A, wdc):
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


def HNX_modularity(HG, A, wdc=linear):
    Pr = compute_partition_probas(HG, A)
    return EdgeContribution(HG, A, wdc) - DegreeTax(HG, Pr, wdc)

################################################################################

# 2-section igraph from HG


def HNX_2section(HG):
    s = []
    for e in HG.edges:
        E = HG.edges[e]
        # random-walk 2-section (preserve nodes' weighted degrees)
        try:
            w = HG.edges[e].weight / (len(E) - 1)
        except:
            w = 1 / (len(E) - 1)
        s.extend([(k[0], k[1], w) for k in itertools.combinations(E, 2)])
    G = ig.Graph.TupleList(s, weights=True).simplify(combine_edges='sum')
    return G

################################################################################

def HNX_Kumar(HG, delta=.01):

    # weights will be modified -- store initial weights
    W = [e.weight for e in HG.edges()]
    # build graph
    G = HNX_2section(HG)
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
        for i in HG.edges:
            e = HG.edges[i]
            reweight = sum([1 / (1 + HG.size(e, c)) for c in CH]) * (HG.size(e) + len(CH)) / HG.number_of_edges()
            diff = max(diff, 0.5 * abs(e.weight - reweight))
            e.weight = 0.5 * e.weight + 0.5 * reweight
        # re-run louvain
        # build graph
        G = HNX_2section(HG)
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
    return {v['name']: v['part'] for v in G.vs}

################################################################################

# compute change in edge contribution --
# partition P, node v going from P[a] to P[b]


def DeltaEC(HG, P, v, a, b, wdc):
    Pm = P[a] - {v}
    Pn = P[b].union({v})
    ec = 0
    for e in list(HG.nodes[v].memberships):
        d = HG.size(e)
        w = HG.edges[e].weight
        ec += w * (wdc(d, HG.size(e, Pm)) + wdc(d, HG.size(e, Pn))
                   - wdc(d, HG.size(e, P[a])) - wdc(d, HG.size(e, P[b])))
    return ec / HG.total_weight

# exp. part of binomial pmf


def bin_ppmf(d, c, p):
    return p**c * (1 - p)**(d - c)

# compute change in degree tax --
# partition P (list), node v going from P[a] to P[b]
def DeltaDT(HG, P, v, a, b, wdc):

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
            x += HG.bin_coef[(d, c)] * wdc(d, c) * (bin_ppmf(d, c, voln) + bin_ppmf(d, c, volm)
                                                    - bin_ppmf(d, c, vola) - bin_ppmf(d, c, volb))
        DT += x * HG.d_weights[d]
    return DT / HG.total_weight

# simple H-based algorithm --
# try moving nodes between communities to optimize qH
# requires L: initial non-trivial partition


def HNX_LastStep(HG, L, wdc=linear, delta=.01):
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
                        M.append(DeltaEC(HG, A, v, c, i, wdc) - DeltaDT(HG, A, v, c, i, wdc))
                i = s[np.argmax(M)]
                if c != i:
                    A[c] = A[c] - {v}
                    A[i] = A[i].union({v})
                    D[v] = i
        Pr = compute_partition_probas(HG, A)
        q2 = EdgeContribution(HG, A, wdc) - DegreeTax(HG, Pr, wdc)
        if (q2 - qH) < delta:
            break
        qH = q2
    return [a for a in A if len(a) > 0]

################################################################################
