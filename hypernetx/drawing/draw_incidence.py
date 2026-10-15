import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection, EllipseCollection
import matplotlib.patheffects as path_effects

from .util import (
    inflate_kwargs,
    transpose_inflated_kwargs,
    inflate,
    get_frozenset_label,
)
from .rubber_band import add_edge_defaults


def draw_incidence_upset(
    H,
    ax=None,
    node_radius=None,
    edge_order=None,
    node_order=None,
    node_labels=None,
    edge_labels=None,
    with_node_labels=True,
    with_edge_labels=True,
    fill_edges=False,
    fill_edge_alpha=-0.5,
    edges_kwargs={},
    nodes_kwargs={},
    edge_labels_kwargs={},
    node_labels_kwargs={},
    edge_labels_on_axis=True,
    node_labels_on_axis=False,
):
    """
    Draw a hypergraph as an incidence matrix within a Matplotlib figure

    This will draw a colorful visualization inspired by the UpSet [1] approach for
    drawing set systems.

    Node and edge visual encodings are specified the same way as hypernetx.draw.

    The default layout algorithm is based on the nx.spectral_order ordering of the
    nodes in the bipartite representation of the hypergraph. You may specifiy custom
    node or edge orderings, e.g. if nodes have timestamps they should be ordered by.

    Edges are assigned a coordinate on the x-axis, and nodes are assigned a
    coordinate on the y-axis. To reverse this, use hypernetx.Hypergraph.dual() on
    the input, and swap arguments as appropriate.

    References
    ----------
    [1] Lex, Alexander, et al. "UpSet: visualization of intersecting sets." IEEE
        transactions on visualization and computer graphics 20.12 (2014): 1983-1992.

    Parameters
    ----------
    H: hnx.Hypergraph
        the entity to be drawn
    ax: Axis
        matplotlib axis on which the plot is rendered
    edge_order: list
        the order to show edges (on the x-axis), overriding the layout algorithm
    node_order: list
        the order to show nodes (on the y-axis), overriding the layout algorithm
    node_labels: function, dict, list, string, float, or int
        determines the text drawn for nodes
    edge_labels: function, dict, list, string, float, or int
        determines the text drawn for edges
    with_node_labels: bool
        set to True to disable showing labels for nodes
    with_edge_labels: bool
        set to True to disable showing labels for edges
    fill_edges: bool
        set to True to fill set the facecolor of edges to a lighter version of the edgecolor if no facecolor is otherwise specified
    fill_edge_alpha: float
        amount to add to the alpha channel when filling edges. Should be between -1 and 0, causing a decrease in alpha
    edges_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for edges
    nodes_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for nodes
    edge_labels_kwargs: dict
        keyword arguments passed to matplotlib.annotate for edge labels
    node_labels_kwargs: dict
        keyword arguments passed to matplotlib.annotate for node labels
    edge_labels_on_axis: bool
        show edge label on the matplotlib x-axis instead of next to the edge
    node_labels_on_axis: bool
        show node label on the matplotlib y-axis instead of next to the edge
    """

    edges_kwargs = add_edge_defaults(H, edges_kwargs)

    default_node_color = "black"

    ax = ax or plt.gca()

    if node_order is None or edge_order is None:
        order = nx.spectral_ordering(H.bipartite(), seed=1234567890)

        if node_order is None:
            node_order = list(filter(H.nodes.__contains__, order))

        if edge_order is None:
            edge_order = list(filter(H.edges.__contains__, order))

    def get_pos(order):
        return {v: i for i, v in enumerate(order)}

    def get_extent(view, pos):
        return {
            v: (min(map(pos.get, view[v])), max(map(pos.get, view[v]))) for v in view
        }

    node_pos = get_pos(node_order)
    edge_pos = get_pos(edge_order)

    node_extent = get_extent(H.nodes, edge_pos)
    edge_extent = get_extent(H.edges, node_pos)

    def create_collection(elements, points, kwargs={}):
        return LineCollection(
            points,
            path_effects=[path_effects.Stroke(capstyle="round")],
            **inflate_kwargs(elements, kwargs)
        )

    node_lines = create_collection(
        H.nodes,
        [
            ((node_extent[v][0], node_pos[v]), (node_extent[v][1], node_pos[v]))
            for v in H.nodes
        ],
        {"colors": nodes_kwargs.get("edgecolors", default_node_color)},
    )

    node_edgecolors = dict(zip(H.nodes, node_lines.get_colors()))

    r = 1 / 3
    theta = np.linspace(0, np.pi, 50)
    half_circle = np.vstack([r * np.cos(theta), r * np.sin(theta)]).T

    def make_points(x, y1, y2):
        return np.vstack(
            [-half_circle - [0, r] + [x, y1], half_circle + [0, r] + [x, y2]]
        )

    edge_lines = PolyCollection(
        [make_points(edge_pos[v], *edge_extent[v]) for v in H.edges],
        **inflate_kwargs(H.edges, edges_kwargs)
    )

    if fill_edges:
        color = edge_lines.get_edgecolors() + np.array([0, 0, 0, fill_edge_alpha])
        edge_lines.set_facecolors(color)

    ax.add_collection(edge_lines)

    node_facecolors = dict(
        zip(
            H.nodes,
            inflate(H.nodes, nodes_kwargs.get("facecolors", default_node_color)),
        )
    )

    incidences = [(v, e) for v in H for e in H.nodes[v]]

    offsets = [(edge_pos[e], node_pos[v]) for v, e in incidences]

    node_radius_dict = {}
    rmax = 1

    if node_radius is not None:
        node_radius_dict = dict(zip(H.nodes, inflate(H.nodes, node_radius)))
        rmax = max(node_radius_dict.values())

    sizes = [1.5 * r * node_radius_dict.get(v, 1) / rmax for v, _ in incidences]

    circles = EllipseCollection(
        widths=sizes,
        heights=sizes,
        angles=0,
        units="xy",
        offsets=offsets,
        transOffset=ax.transData,
        facecolors=[node_facecolors[v] for v, e in incidences],
        edgecolors=(
            [node_edgecolors[v] for v, e in incidences]
            if "edgecolors" in nodes_kwargs
            else None
        ),
    )
    ax.add_collection(circles)

    def clear_y_axis():
        ax.set_yticks([], [])
        ax.spines["left"].set_visible(False)

    def clear_x_axis():
        ax.set_xticks([], [])
        ax.spines["bottom"].set_visible(False)

    # node labels

    if with_node_labels:
        node_labels_inflated = (
            list(H.nodes) if node_labels is None else inflate(H.nodes, node_labels)
        )

        if node_labels_on_axis:
            ax.set_yticks([node_pos[v] for v in H.nodes], node_labels_inflated)
        else:
            clear_y_axis()

            if len(node_labels_kwargs) > 0:
                node_labels_kwargs_inflated = transpose_inflated_kwargs(
                    inflate_kwargs(H.nodes, node_labels_kwargs)
                )
            else:
                node_labels_kwargs_inflated = [{}] * len(H.nodes)

            for v, s, kwargs in zip(
                H.nodes, node_labels_inflated, node_labels_kwargs_inflated
            ):
                xy = np.array([node_extent[v][0], node_pos[v]])
                ax.annotate(
                    s,
                    np.array([-r, 0]) + xy,
                    **{
                        "xytext": (-2, 0),
                        "textcoords": "offset pixels",
                        "ha": "right",
                        "va": "center",
                        **kwargs,
                    }
                )
    else:
        clear_y_axis()

    # edge labels

    if with_edge_labels:
        colors = edge_lines.get_edgecolors()
        edge_labels_kwargs_inflated = inflate_kwargs(
            H.edges, {"color": colors, **edge_labels_kwargs}
        )

        edge_labels_inflated = (
            list(H.edges) if edge_labels is None else inflate(H.edges, edge_labels)
        )

        if edge_labels_on_axis:
            ax.set_xticks([edge_pos[e] for e in H.edges], edge_labels_inflated)
        else:
            clear_x_axis()

            for v, s, kwargs in zip(
                H.edges,
                edge_labels_inflated,
                transpose_inflated_kwargs(edge_labels_kwargs_inflated),
            ):
                xy = np.array([edge_pos[v], edge_extent[v][0]])
                ax.annotate(
                    s,
                    np.array([0, -2 * r]) + xy,
                    **{
                        "ha": "center",
                        "va": "top",
                        "xytext": (0, -2),
                        "textcoords": "offset pixels",
                        **kwargs,
                    }
                )
    else:
        clear_x_axis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.add_collection(node_lines)

    ax.axis("equal")
    ax.autoscale_view()
