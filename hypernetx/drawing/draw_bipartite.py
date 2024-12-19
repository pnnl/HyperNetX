import numpy as np
import networkx as nx

import hypernetx as hnx


def rebalance(B, pos, side=None):
    new_pos = {}

    if side is None:
        side = B

    for u in side:
        x = pos[u][0]
        y = np.mean([pos[v][1] for v in B[u]])
        new_pos[u] = (x, y)

    return {**pos, **new_pos}


def bipartite_layout(B, left_side, width=1):
    B = B.to_undirected()
    left_side = set(left_side)
    right_side = set()

    pos = {}
    n = 0
    m = 0

    for v in nx.spectral_ordering(B):
        if v in left_side:
            pos[v] = (0, m)
            m += 1
        else:
            right_side.add(v)
            pos[v] = (width, n)
            n += 1

    if len(left_side) < len(right_side):
        pos = rebalance(B, pos, left_side)
    elif len(left_side) > len(right_side):
        pos = rebalance(B, pos, right_side)

    return pos


def draw_bipartite_using_euler(
    H, node_order=None, edge_order=None, edge_labels_kwargs={}, **kwargs
):
    B = H.bipartite()
    pos = bipartite_layout(B, H.edges, width=0.5 * max(len(H.nodes), len(H.edges)))

    return hnx.drawing.draw(
        H,
        pos=pos,
        with_additional_edges=B,
        contain_hyper_edges=True,
        edge_labels_on_edge=False,
        edge_labels_kwargs={
            "xytext": (-10, 0),
            "textcoords": "offset points",
            **edge_labels_kwargs,
        },
        **kwargs
    )
