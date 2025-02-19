import numpy as np
import networkx as nx

import hypernetx as hnx


def rebalance(B, pos, side):
    new_pos = {}

    y = -1
    for u in side:
        x = pos[u][0]
        y = max(y + 1, min([pos[v][1] for v in B[u]]))
        new_pos[u] = (x, y)

    return {**pos, **new_pos}


def sort_left_by_right(B, left, right):
    pos = {v: i for i, v in enumerate(right)}
    print(left, right)

    return sorted(left, key=lambda u: np.mean([pos[v] for v in B[u]]))


def bipartite_layout(B, left_side, right_side, width=1):

    pos = {}

    for i, v in enumerate(left_side):
        pos[v] = (0, i)

    for i, v in enumerate(right_side):
        pos[v] = (width, i)

    if len(left_side) < len(right_side):
        pos = rebalance(B, pos, left_side)
        pos = rebalance(B, pos, right_side)
    elif len(left_side) > len(right_side):
        pos = rebalance(B, pos, right_side)
        pos = rebalance(B, pos, left_side)

    return pos


def draw_bipartite_using_euler(
    H, pos=None, node_order=None, edge_order=None, edge_labels_kwargs={}, **kwargs
):
    """
    Draw a hypergraph as a two column bipartite graph using hypernetx.draw.

    This function calculates the x- and y-coordinates of nodes and edges by placing
    edges in the left column and nodes in the right column. The ordering of edges
    and nodes is determined by default using the networkx.spectral_order method
    on the bipartite representation of the hypergraph. Node and edge order can
    also be specified by the user. If one or the other is specified, then the
    unspecified column (nodes or edges) are sorted using the barycenter method.
    Additional minor adjustment of node and edge height is performed to improve
    readability.

    Additional encoding kwargs are passed through to the hypernetx.draw method.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    pos: dict
        the positioning of the hypergraph
    node_order: list
        specify to override the order of the nodes in the drawing
    edge_order: list
        specify to override the order of the edges in the drawing
    """

    B = H.bipartite().to_undirected()
    if pos is None:
        if node_order is None and edge_order is not None:
            node_order = sort_left_by_right(B, H.nodes, edge_order)
        elif node_order is not None and edge_order is None:
            edge_order = sort_left_by_right(B, H.edges, node_order)
        elif node_order is None and edge_order is None:
            order = nx.spectral_ordering(B, seed=1234567890)
            node_order = list(filter(H.nodes.__contains__, order))
            edge_order = list(filter(H.edges.__contains__, order))

        pos = bipartite_layout(
            B, edge_order, node_order, width=0.5 * max(len(H.nodes), len(H.edges))
        )

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
