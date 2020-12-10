# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

from hypernetx import Hypergraph
from .util import get_frozenset_label, get_set_layering, inflate_kwargs, transpose_inflated_kwargs

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection, CircleCollection

import networkx as nx

from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

# increases the default figure size to 8in square.
plt.rcParams['figure.figsize'] = (8, 8)

N_CONTROL_POINTS = 24

theta = np.linspace(0, 2 * np.pi, N_CONTROL_POINTS + 1)[:-1]

cp = np.vstack((np.cos(theta),
                np.sin(theta))).T


def layout_node_link(H, layout=nx.spring_layout, **kwargs):
    '''
    Helper function to use a NetwrokX-like graph layout algorithm on a Hypergraph

    The hypergraph is converted to a bipartite graph, allowing the usual graph layout
    techniques to be applied. 

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    layout: function
        the layout algorithm which accepts a NetworkX graph and keyword arguments
    kwargs: dict
        Keyword arguments are passed through to the layout algorithm

    Returns
    -------
    dict
        mapping of node and edge positions to R^2        
    '''
    return layout(H.bipartite(), **kwargs)

def get_default_radius(H, pos):
    '''
    Calculate a reasonable default node radius

    This function iterates over the hyper edges and finds the most distant
    pair of points given the positions provided. Then, the node radius is a fraction
    of the median of this distance take across all hyper-edges.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2

    Returns
    -------
    float
        the recommended radius

    '''
    if len(H) > 1:
        return .0125 * np.median([
            pdist(np.vstack(list(map(pos.get, H.nodes)))).max()
            for nodes in H.edges()
        ])
    return 1


def draw_hyper_edge_labels(H, polys, labels={}, ax=None, **kwargs):
    '''
    Draws a label on the hyper edge boundary.

    Should be passed Matplotlib PolyCollection representing the hyper-edges, see
    the return value of draw_hyper_edges.

    The label will be draw on the least curvy part of the polygon, and will be
    aligned parallel to the orientation of the polygon where it is drawn.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    polys: PolyCollection
        collection of polygons returned by draw_hyper_edges
    labels: dict
        mapping of node id to string label
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        Keyword arguments are passed through to Matplotlib's annotate function.

    '''
    ax = ax or plt.gca()

    params = transpose_inflated_kwargs(
        inflate_kwargs(H.edges, kwargs)
    )

    for edge, path, params in zip(H.edges, polys.get_paths(), params):
        s = labels.get(edge, edge)

        # calculate the xy location of the annotation
        # this is the midpoint of the pair of adjacent points the most distant
        d = ((path.vertices[:-1] - path.vertices[1:])**2).sum(axis=1)
        i = d.argmax()

        x1, x2 = path.vertices[i:i + 2]
        x, y = (x2 - x1)
        theta = 360 * np.arctan2(y, x) / (2 * np.pi)
        theta = (theta + 360) % 360

        while theta > 90:
            theta -= 180

        # the string is a comma separated list of the edge uid
        ax.annotate(s, (x1 + x2) / 2,
                    rotation=theta,
                    ha='center', va='center',
                    **params)


def layout_hyper_edges(H, pos, node_radius={}, dr=None):
    '''
    Draws a convex hull for each edge in H.

    Position of the nodes in the graph is specified by the position dictionary,
    pos. Convex hulls are spaced out such that if one set contains another, the
    convex hull will surround the contained set. The amount of spacing added
    between hulls is specified by the parameter, dr.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    dr: float
        the spacing between concentric rings
    ax: Axis
        matplotlib axis on which the plot is rendered

    Returns
    -------
    dict
        A mapping from hyper edge ids to paths (Nx2 numpy matrices)
    '''

    if len(node_radius):
        r0 = min(node_radius.values())
    else:
        r0 = get_default_radius(H, pos)

    dr = dr or r0

    levels = get_set_layering(H)

    radii = {v: {v: i for i, v in enumerate(sorted(e, key=levels.get))}
             for v, e in H.dual().edges.elements.items()}

    def get_padded_hull(uid, edge):
        # make sure the edge contains at least one node
        if len(edge):
            points = np.vstack([cp * (node_radius.get(v, r0) + dr * (2 + radii[v][uid])) + pos[v]
                                for v in edge])
        # if not, draw an empty edge centered around the location of the edge node (in the bipartite graph)
        else:
            points = 4 * r0 * cp + pos[uid]

        hull = ConvexHull(points)

        return hull.points[hull.vertices]

    return [
        get_padded_hull(uid, list(H.edges[uid]))
        for uid in H.edges
    ]


def draw_hyper_edges(H, pos, ax=None, node_radius={}, dr=None, **kwargs):
    '''
    Draws a convex hull around the nodes contained within each edge in H

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    dr: float
        the spacing between concentric rings
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments, e.g., linewidth, facecolors, are passed through to the PolyCollection constructor

    Returns
    -------
    PolyCollection
        a Matplotlib PolyCollection that can be further styled
    '''
    points = layout_hyper_edges(H, pos, node_radius=node_radius, dr=dr)

    polys = PolyCollection(points, **inflate_kwargs(H.edges, kwargs))

    (ax or plt.gca()).add_collection(polys)

    return polys


def draw_hyper_nodes(H, pos, node_radius={}, r0=None, ax=None, **kwargs):
    '''
    Draws a circle for each node in H.

    The position of each node is specified by the a dictionary/list-like, pos,
    where pos[v] is the xy-coordinate for the vertex. The radius of each node
    can be specified as a dictionary where node_radius[v] is the radius. If a
    node is missing from this dictionary, or the node_radius is not specified at
    all, a sensible default radius is chosen based on distances between nodes
    given by pos.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    r0: float
        minimum distance that concentric rings start from the node position
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments, e.g., linewidth, facecolors, are passed through to the PolyCollection constructor

    Returns
    -------
    PolyCollection
        a Matplotlib PolyCollection that can be further styled
    '''

    ax = ax or plt.gca()

    r0 = r0 or get_default_radius(H, pos)

    points = [node_radius.get(v, r0) * cp + pos[v] for v in H.nodes]

    kwargs.setdefault('facecolors', 'black')

    circles = PolyCollection(points, **inflate_kwargs(H, kwargs))

    ax.add_collection(circles)

    return circles


def draw_hyper_labels(H, pos, node_radius={}, ax=None, labels={}, **kwargs):
    '''
    Draws text labels for the hypergraph nodes.

    The label is drawn to the right of the node. The node radius is needed (see
    draw_hyper_nodes) so the text can be offset appropriately as the node size
    changes.

    The text label can be customized by passing in a dictionary, labels, mapping
    a node to its custom label. By default, the label is the string
    representation of the node.

    Keyword arguments are passed through to Matplotlib's annotate function.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    pos: dict
        mapping of node and edge positions to R^2
    node_radius: dict
        mapping of node to R^1 (radius of each node)
    ax: Axis
        matplotlib axis on which the plot is rendered
    labels: dict
        mapping of node to text label
    kwargs: dict
        keyword arguments passed to matplotlib.annotate

    '''
    ax = ax or plt.gca()

    params = transpose_inflated_kwargs(
        inflate_kwargs(H.nodes, kwargs)
    )

    for v, v_kwargs in zip(H.nodes, params):
        xy = np.array([node_radius.get(v, 0), 0]) + pos[v]
        ax.annotate(labels.get(v, v), xy,
                    **{k: (d[v] if hasattr(d, '__getitem__') and type(d) not in {str, tuple} else d)
                       for k, d in kwargs.items()})


def draw(H,
         pos=None,
         with_color=True,
         with_node_counts=False,
         with_edge_counts=False,
         layout=nx.spring_layout,
         layout_kwargs={},
         ax=None,
         node_radius=None,
         edges_kwargs={},
         nodes_kwargs={},
         edge_labels={},
         edge_labels_kwargs={},
         node_labels={},
         node_labels_kwargs={},
         with_edge_labels=True,
         with_node_labels=True,
         label_alpha=.35,
         return_pos=False
         ):
    '''
    Draw a hypergraph as a Matplotlib figure

    By default this will draw a colorful "rubber band" like hypergraph, where
    convex hulls represent edges and are drawn around the nodes they contain.

    This is a convenience function that wraps calls with sensible parameters to
    the following lower-level drawing functions:

    * draw_hyper_edges,
    * draw_hyper_edge_labels,
    * draw_hyper_labels, and
    * draw_hyper_nodes

    The default layout algorithm is nx.spring_layout, but other layouts can be
    passed in. The Hypergraph is converted to a bipartite graph, and the layout
    algorithm is passed the bipartite graph.

    If you have a pre-determined layout, you can pass in a "pos" dictionary.
    This is a dictionary mapping from node id's to x-y coordinates. For example:

        >>> pos = {
        >>> 'A': (0, 0),
        >>> 'B': (1, 2),
        >>> 'C': (5, -3)
        >>> }

    will position the nodes {A, B, C} manually at the locations specified. The
    coordinate system is in Matplotlib "data coordinates", and the figure will
    be centered within the figure.

    By default, this will draw in a new figure, but the axis to render in can be
    specified using :code:`ax`.

    This approach works well for small hypergraphs, and does not guarantee
    a rigorously "correct" drawing. Overlapping of sets in the drawing generally
    implies that the sets intersect, but sometimes sets overlap if there is no
    intersection. It is not possible, in general, to draw a "correct" hypergraph
    this way for an arbitrary hypergraph, in the same way that not all graphs
    have planar drawings.    

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn 
    pos: dict
        mapping of node and edge positions to R^2
    with_color: bool
        set to False to disable color cycling of edges
    with_node_counts: bool
        set to True to label collapsed nodes with number of elements
    with_edge_counts: bool
        set to True to label collapsed edges with number of elements
    layout: function
        layout algorithm to compute
    layout_kwargs: dict
        keyword arguments passed to layout function
    ax: Axis
        matplotlib axis on which the plot is rendered
    edges_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for edges
    node_radius: None, int, float, or dict
        radius of all nodes, or dictionary of node:value; the default (None) calculates radius based on number of collapsed nodes; reasonable values range between 1 and 3
    nodes_kwargs: dict
        keyword arguments passed to matplotlib.collections.PolyCollection for nodes
    edge_labels_kwargs: dict
        keyword arguments passed to matplotlib.annotate for edge laberls
    node_labels_kwargs: dict
        keyword argumetns passed to matplotlib.annotate for node labels
    with_edge_labels: bool
        set to False to make edge labels invisible
    with_node_labels: bool
        set to False to make node labels invisible
    label_alpha: float
        the transparency (alpha) of the box behind text drawn in the figure
    '''

    ax = ax or plt.gca()

    if pos is None:
        pos = layout_node_link(H, layout=layout, **layout_kwargs)

    r0 = get_default_radius(H, pos)
    a0 = np.pi * r0**2

    def get_node_radius(v):
        if node_radius is None:
            return np.sqrt(a0 * (len(v) if type(v) == frozenset else 1) / np.pi)
        elif hasattr(node_radius, 'get'):
            return node_radius.get(v, 1) * r0
        return node_radius * r0

    # guarantee that node radius is a dictionary mapping nodes to values
    node_radius = {
        v: get_node_radius(v)
        for v in H.nodes
    }

    # for convenience, we are using setdefault to mutate the argument
    # however, we need to copy this to prevent side-effects
    edges_kwargs = edges_kwargs.copy()
    edges_kwargs.setdefault('edgecolors', plt.cm.tab10(np.arange(len(H.edges)) % 10))
    edges_kwargs.setdefault('facecolors', 'none')

    polys = draw_hyper_edges(H, pos,
                             node_radius=node_radius,
                             ax=ax,
                             **edges_kwargs
                             )

    if with_edge_labels:
        labels = get_frozenset_label(
            H.edges,
            count=with_edge_counts,
            override=edge_labels
        )

        draw_hyper_edge_labels(H, polys,
                               color=edges_kwargs['edgecolors'],
                               backgroundcolor=(1, 1, 1, label_alpha),
                               labels=labels,
                               ax=ax,
                               **edge_labels_kwargs
                               )

    if with_node_labels:
        labels = get_frozenset_label(
            H.nodes,
            count=with_node_counts,
            override=node_labels
        )

        draw_hyper_labels(H, pos,
                          node_radius=node_radius,
                          labels=labels,
                          ax=ax,
                          va='center',
                          xytext=(5, 0),
                          textcoords='offset points',
                          backgroundcolor=(1, 1, 1, label_alpha),
                          **node_labels_kwargs
                          )

    draw_hyper_nodes(H, pos,
                     node_radius=node_radius,
                     ax=ax,
                     **nodes_kwargs
                     )

    if len(H.nodes) == 1:
        x, y = pos[list(H.nodes)[0]]
        s = 20

        ax.axis([x - s, x + s, y - s, y + s])
    else:
        ax.axis('equal')

    ax.axis('off')
    if return_pos:
        return pos
