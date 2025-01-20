import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

from .util import inflate_kwargs, transpose_inflated_kwargs, inflate, get_frozenset_label


def draw_incidence_upset(
    H,
    ax=None,
    node_labels=None,
    edge_labels=None,
    with_node_labels=True,
    with_edge_labels=True,
    with_node_counts=False,
    with_edge_counts=False,
    edge_order=None,
    node_order=None,
    edges_kwargs={},
    nodes_kwargs={},
    edge_labels_kwargs={},
    node_labels_kwargs={},
    edge_labels_on_axis=True,
    node_labels_on_axis=False,
    edge_epsilon=0.05,
):
    if 'facecolors' in edges_kwargs:
        edges_kwargs = dict(edges_kwargs)
        edges_kwargs['colors'] = edges_kwargs['facecolors']
        del edges_kwargs['facecolors']

    edges_kwargs.setdefault('colors', plt.cm.tab10(np.arange(len(H.edges)) % 10) + np.array([0, 0, 0, -.5]))

    default_edge_color = 'lightgray'
    default_node_color = 'black'
    default_edge_width = 15
    node_radius = default_edge_width / 3

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
            path_effects=[path_effects.Stroke(capstyle='round')],
            **inflate_kwargs(elements, kwargs)
        )

    node_lines = create_collection(
        H.nodes,
        [
            ((node_extent[v][0], node_pos[v]), (node_extent[v][1], node_pos[v]))
            for v in H.nodes
        ],
        {'colors': nodes_kwargs.get('edgecolors', default_node_color)},
    )

    node_edgecolors = dict(zip(H.nodes, node_lines.get_colors()))

    edge_lines = create_collection(
        H.edges,
        [
            (
                (edge_pos[v], edge_extent[v][0] - edge_epsilon),
                (edge_pos[v], edge_extent[v][1] + edge_epsilon),
            )
            for v in H.edges
        ],
        {'linewidth': default_edge_width, 'color': default_edge_color, **edges_kwargs},
    )

    incidences = [(v, e) for v in H for e in H.nodes[v]]

    ax.add_collection(edge_lines)

    node_facecolors = dict(zip(
        H.nodes,
        inflate(H.nodes, nodes_kwargs.get('facecolors', default_node_color))
    ))

    ax.scatter(
        [edge_pos[e] for v, e in incidences],
        [node_pos[v] for v, e in incidences],
        s=np.pi * node_radius**2,
        zorder=3,
        facecolors=[node_facecolors[v] for v, e in incidences],
        edgecolors=[node_edgecolors[v] for v, e in incidences] if 'edgecolors' in nodes_kwargs else None
    )

    def clear_y_axis():
        ax.set_yticks([], [])
        ax.spines['left'].set_visible(False)


    def clear_x_axis():
        ax.set_xticks([], [])
        ax.spines['bottom'].set_visible(False)

    # node labels
    
    if with_node_labels:
        node_labels_inflated = H.nodes 
        if with_node_counts:
            node_labels = get_frozenset_label(H.nodes, count=True, override=node_labels)
        elif node_labels is not None:
            node_labels = inflate(H.nodes, node_labels)

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

            for v, s, kwargs in zip(H.nodes, node_labels_inflated, node_labels_kwargs_inflated):
                xy = (node_extent[v][0], node_pos[v])
                ax.annotate(
                    s,
                    xy,
                    ha='right',
                    va='center',
                    xytext=(-default_edge_width / 2 - 4, 0),
                    textcoords='offset pixels',
                    **kwargs
                )
    else:
        clear_y_axis()

    # edge labels

    if with_edge_labels:
        edge_labels_inflated = H.edges 
        if with_edge_counts:
            edge_labels_inflated = get_frozenset_label(H.edges, count=True, override=edge_labels)
        elif edge_labels is not None:
            edge_labels_inflated = inflate(H.edges, edge_labels)
            
        if edge_labels_on_axis:
            ax.set_xticks([edge_pos[e] for e in H.edges], edge_labels_inflated)
        else:
            clear_x_axis()

            if len(edge_labels_kwargs) > 0:
                edge_labels_kwargs_inflated = transpose_inflated_kwargs(
                    inflate_kwargs(H.edges, edge_labels_kwargs)
                )
            else:
                edge_labels_kwargs_inflated = [{}] * len(H.edges)

            for v, s, kwargs in zip(H.edges, edge_labels_inflated, edge_labels_kwargs_inflated):
                xy = (edge_pos[v], edge_extent[v][0])
                ax.annotate(
                    s,
                    xy,
                    ha='center',
                    va='top',
                    xytext=(0, -default_edge_width / 2 - 6),
                    textcoords='offset pixels',
                    **kwargs
                )
    else:
        clear_x_axis()

        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.add_collection(node_lines)
    ax.autoscale_view()
