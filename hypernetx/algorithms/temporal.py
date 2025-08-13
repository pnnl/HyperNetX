import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import hypernetx as hnx
from hypernetx.drawing import draw_storyline as ds

import networkx as nx

from collections import defaultdict

NODE_COLOR = 'black'
NODE_ENDPOINT_COLOR = 'red'
NODE_LINEWIDTH = 1
NODE_PATH_LINEWIDTH = 3
HYPER_EDGE_FACECOLOR = 'white'
HYPER_EDGE_EDGECOLOR = 'darkgray'


def get_edge_pos(edge_order, edge_pos):
    """
    Helper function to convert edge_order into edge_pos

    If edge_order is not None, each edge is assigned a position of
        0, 1, ..., n - 1

    If edge_pos is not None, edge_pos returned

    Parameters
    ----------
    edge_pos: dict
        mapping of hyper edges to an int/float representing the edge timestamp
    edge_order: list
        list specifying the temporal order of edges

    Returns
    ----------
    dict
        mapping of hyper edges to a numerical/temporal value

    """

    assert (edge_pos is None) ^ (
        edge_order is None
    ), 'Exactly one of edge_pos and edge_order must be specified'

    if edge_pos is None:
        edge_pos = {v: i for i, v in enumerate(edge_order)}

    return edge_pos


def create_temporal_incidence_graph(H, edge_pos=None, edge_order=None, method='quick'):
    """
    Convert an edge ordered hypergraph into a temporal incidence graph

    This method constructs and weights a temporal incidence graph that is then
    used to compute temporal shortest hypergraph paths. Nodes in this graph
    are either incidences, i.e. (hyper edge, node) tuples, or hyper edges.

    Edges are created between temporally adjacent incidences sharing the same
    hyper node, or between incidences sharing the same hyper edge.

    Exactly of edge_pos or edge_order must be provided. If edge_order is
    provided, edge_pos is inferred from this.

    Parameters
    ----------
    H: networkx.DiGraph
        the edge ordered Hypergraph
    edge_pos: dict
        mapping of hyper edges to an int/float representing the edge timestamp
    edge_order: list
        list specifying the temporal order of edges
    method: str or func
        the weighting method; if a string is passed, should be either 'quick'
        or 'fast'. If a func is passed, that function will find the weight
        of each edge in the constructed graph.

    Returns
    ----------
    networkx.DiGraph
        the constructed graph

    """

    edge_pos = get_edge_pos(edge_order, edge_pos)

    if method == 'quick':

        def weight_func(u, v):
            if u in edge_pos or v in edge_pos:
                return 0

            return edge_pos[v[0]] - edge_pos[u[0]]

    elif method == 'short':

        def weight_func(u, v):
            if u in edge_pos or v in edge_pos:
                return 1
            return 0

    else:
        assert True, 'Method must be "quick", "fast", or a function'
        weight_func = method

    G = nx.DiGraph()

    for v in H.nodes():
        edges = sorted(H.nodes[v], key=edge_pos.get)
        for e in edges:
            G.add_edge((e, v), e, weight=weight_func((e, v), e))
            G.add_edge(e, (e, v), weight=weight_func(e, (e, v)))

        # temporal/directed edges connecting incidences
        for e1, e2 in zip(edges[:-1], edges[1:]):
            G.add_edge((e1, v), (e2, v), weight=weight_func((e1, v), (e2, v)))

    return G


def default_layout(G):
    """
    Default layout function used by `draw_temporal_incidence_graph`

    Attempts to render using GraphViz layout. If that fails, e.g. due to the
    package dependencies not being installed, defaults to using the NetworkX
    `kamada_kawai_layout`.

    Parameters
    ----------
    G: networkx.DiGraph
        the graph to be positioned

    Returns
    ----------
    dict
        mapping of graph nodes to (x, y) coordinates
    """

    try:
        return nx.nx_agraph.graphviz_layout(G, prog='neato')
    except:
        return nx.kamada_kawai_layout(G, weight=None)


def draw_temporal_incidence_graph(
    G,
    *,
    pos=None,
    layout=default_layout,
    path=None,
    show_weights=True,
    show_zero_weight=False,
    node_color=NODE_COLOR,
    node_endpoint_color=NODE_ENDPOINT_COLOR,
    width=NODE_LINEWIDTH,
    path_width=NODE_PATH_LINEWIDTH,
    hyper_edge_color=HYPER_EDGE_EDGECOLOR,
    hyper_edge_font_color='black',
    weight_font_color='red',
    incidence_alpha=0.85,
    sm_font=8,
    md_font=10,
    lg_font=12,
    hyper_edge_node_size=300,
    incidence_node_size=500,
    ax=None
):
    """
    Node-link visualization of temporal incidence graph

    This is a visualization of the underlying graph structure used to solve
    the temporal hypergraph shortest path problem. The intended use of this
    visualization is to show the edge weights, direction, incidence
    information, and path to help illustrate the concept of the temporal
    hypergraph path.

    Parameters
    ----------
    G: networkx.DiGraph
        the temporal incidence graph to be drawn
    pos: dict
        the location of the nodes to be drawn
    layout: func
        the algorithm to layout the nodes if pos=None
    path: list
        the list of incidences to highlight in the drawing
    show_weights: Boolean
        if True, shows the edge weights in the drawing
    show_zero_weight: Boolean
        if False, weights with value of 0 are not shown
    node_color: color
        color for incidences, drawn as graph nodes
    node_endpoint_color: color
        color to indicate the beginning and end of the path
    width: float
        thickness of edges in the diagram
    path_width: float
        thickness of edges belonging to the path, overrides width
    hyper_edge_color: color
        color of hyper edges, drawn as graph nodes
    hyper_edge_font_color: color
        color of text label for hyper edges
    weight_font_color: color
        color of text labeling edges
    incidence_alpha: float
        transparency of incidence nodes
    sm_font: int
        small font size
    md_font: int
        medium font size
    lg_font: int
        large font size
    hyper_edge_node_size: float
        area of hyper edges drawn as graph nodes
    incidence_node_size
        area of incidences; used only for placement of edge endpoints
    ax: matplotlib.axis.Axis
        axis to render the visualization
    """

    def is_incidence(v):
        return type(v) is tuple

    ax = ax or plt.gca()

    path_nodes = set()
    path_endpoints = set()
    path_edges = set()
    path_hyperedges = set()

    if path is not None:
        path_nodes = set(path)
        path_endpoints = {path[0], path[-1]}

        for (e1, v1), (e2, v2) in zip(path[:-1], path[1:]):
            if e1 == e2:
                path_hyperedges.add(e1)

            if v1 == v2:
                path_edges.add(((e1, v1), (e2, v2)))
            else:
                path_edges.add(((e1, v1), e2))
                path_edges.add((e2, (e2, v2)))

    G.graph = dict(rankdir='LR')

    if pos is None:
        pos = layout(G)

    sizes = [
        incidence_node_size if is_incidence(v) else hyper_edge_node_size for v in G
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[hyper_edge_color if v in path_hyperedges else 'none' for v in G],
        edgecolors=['none' if is_incidence(v) else hyper_edge_color for v in G],
        node_size=sizes,
        ax=ax,
    )

    # edges
    nx.draw_networkx_labels(
        G,
        pos,
        font_color=hyper_edge_font_color,
        # font_weight='bold',
        font_size=lg_font,
        labels={v: v for v in G if not is_incidence(v)},
        ax=ax,
    )

    def get_incidence_colors(v):
        if v in path_endpoints:
            return (node_endpoint_color, node_color)
        elif v in path_nodes:
            return (node_color, 'white')
        return ('white', node_color)

    # incidences
    for v in G:
        if is_incidence(v):
            fc, ec = get_incidence_colors(v)
            ax.annotate(
                ', '.join(map(str, v)),
                pos[v],
                va='center',
                ha='center',
                fontsize=md_font,
                color=ec,
                bbox=dict(facecolor=(*to_rgb(fc), incidence_alpha), edgecolor=ec),
            )

    nx.draw_networkx_edges(
        G,
        pos,
        width=[path_width if e in path_edges else width for e in G.edges()],
        node_size=sizes,
        ax=ax,
    )

    if show_weights:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (u, v): str(d['weight']) if d['weight'] > 0 or show_zero_weight else ''
                for u, v, d in G.edges(data=True)
            },
            font_color=weight_font_color,
            font_weight='bold',
            font_size=sm_font,
            rotate=False,
            ax=ax,
        )


def multi_source_target_dijkstra(G, sources, targets, **kwargs):
    """
    Wrapper for networkx `multi_source_dijkstra` that allows multiple sources or targets

    Calls `networkx.multi_source_dijkstra` in a loop over each target and
    returns the path that minimizes cost. The result will be a path where the
    starting node belongs to one of the specified sources and the ending node
    belongs to one of the specified targets.

    Parameters
    ----------
    G: networkx.DiGraph
        the graph to be computed over

    sources: list
        starting nodes for the shortest path to consider

    targets: list
        ending nodes for the shortest path to consider

    Returns
    ----------
    list, number
        the shortest path and its cost
    """

    return min(
        [
            nx.multi_source_dijkstra(G, sources=sources, target=t, **kwargs)
            for t in targets
        ],
        key=lambda x: x[0],
    )


def find_incidences(H, x):
    """
    A flexible way to retrieve incidences for an object

    If a node is passed, the function returns all incidences containing that
    node. Likewise, if an edge is passed, the function returns all incidences
    containing that edge. If an incidence is passed, that incidence is
    returned in a list.

    This function is used by `temporal_shortest_path` to allow flexible source
    and target input parameters.

    Parameters
    ----------
    H: hypernetx.Hypergraph
        the hypergraph

    x: node, edge, or incidence
        The seed to find related incidences for

    Returns
    ----------
    list
        a list of incidences in the form of (edge, node) tuples
    """

    if type(x) is tuple:
        return [x]
    elif x in H.edges:
        return [(x, v) for v in H.edges[x]]
    elif x in H.nodes:
        return [(e, x) for e in H.nodes[x]]
    return []


def temporal_shortest_path(H, source, target, return_graph=False, **kwargs):
    """
    Solves the temporal hypergraph shrotest path problem

    Calls `create_temporal_incidence_graph` to construct the graph, then calls
    `multi_source_target_dijkstra` to find the shortest path.

    Parameters
    ----------
    H: hypernetx.Hypergraph
        the hypergraph

    source: edge, node, or incidence

    target: edge, node or incidence

    return_graph: Boolean
        if True, the graph create by `create_temporal_incidence_graph` is also returned

    **kwargs:
        additional keyword arguments are passed through `create_temporal_incidence_graph`

    Returns
    ----------
    list, number, [networkx.DiGraph]
        the shortest path and its cost, and optionally, the temporal incidence graph
    """

    G = create_temporal_incidence_graph(H, **kwargs)

    cost, path = multi_source_target_dijkstra(
        G,
        sources=find_incidences(H, source),
        targets=find_incidences(H, target),
        weight='weight',
    )

    # removes the hyper edges from the path so it is just a sequence of incidences

    # Workaround for issue with `HypergraphView.__contains__` in `hyp_view.py` throwing
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    edges = set(H.edges)
    path = [i for i in path if i not in edges]

    if return_graph:
        return cost, path, G

    return cost, path


def encode_path(
    path,
    node_color=NODE_COLOR,
    node_endpoint_color=NODE_ENDPOINT_COLOR,
    node_linewidth=NODE_LINEWIDTH,
    node_path_linewidth=NODE_PATH_LINEWIDTH,
    edge_facecolor=HYPER_EDGE_FACECOLOR,
    edge_edgecolor=HYPER_EDGE_EDGECOLOR,
):
    """
    Create an object that defines `hypernetx.drawing.draw_incidence_*` the
    keyword arguments to highlight the given path.

    By default, the path will have red endpoints and the line representing the
    path will be thicker. Incidences that are part of the path will be filled
    with black, otherwise the fill will be white. Edges that are not part of
    the path will not be filled.

    Usage
    ----------
    The code snippet will find a the temporal shortest path given a hypergraph,
    source, target, and edge order.

        >>> from hypernetx.algorithms.temporal import encode_path, temporal_shortest_path
        >>> from hypernetx import draw_incidence_storyline
        >>> cost, path = temporal_shortest_path(H, source, target, edge_order=edge_order)
        >>> draw_incidence_storyline(H,  edge_order=edge_order, **encode_path(path))

    Parameters
    ----------
    path: list
        list of incidences to highlight in the hypergraph

    node_color: str

    node_endpoint_color: color
        The fill color for the start and end of the path

    node_linewidth: number
        The default width of lines representing nodes

    node_path_linewidth: number
        The widht of liens representing nodes that are in the path, overriding the above parameter

    edge_facecolor: color
        The fill color of edges (and incidences) not in the path

    edge_edgecolor: color
        The stroke color of all edges

    Returns
    ----------
    list
        a list of incidences in the form of (edge, node) tuples
    """

    endpoints = {path[0], path[-1]}
    segments = set(zip(path[:-1], path[1:]))
    edges = set([e1 for (e1, _), (e2, _) in zip(path[:-1], path[1:]) if e1 == e2])

    incidences = [
        j
        for i, j, k in zip(path[:-2], path[1:-1], path[2:])
        if len({i[1], j[1], k[1]}) != 1
    ]

    incidence_facecolor = defaultdict(
        lambda: edge_facecolor,
        dict(zip(incidences, [node_color] * len(incidences))),
    )

    for i in endpoints:
        incidence_facecolor[i] = (
            node_endpoint_color if node_endpoint_color is not None else node_color
        )

    incidence_linewidth = defaultdict(
        lambda: node_linewidth,
        dict(
            zip(
                path,
                [node_path_linewidth / 2] * len(path),
            )
        ),
    )

    return dict(
        edges_kwargs=dict(edgecolor=edge_edgecolor),
        incidences_kwargs=dict(
            facecolor=incidence_facecolor, linewidth=incidence_linewidth
        ),
        segments_kwargs=dict(
            linewidths=lambda seg: (
                node_path_linewidth if seg in segments else node_linewidth
            )
        ),
        fill_edges=defaultdict(lambda: False, {i: True for i in edges}),
    )


def create_temporal_line_graph(H, edge_order=None, edge_pos=None, weight_by_time=True):
    edge_pos = get_edge_pos(edge_order, edge_pos)

    def create_edge(e):
        u, v = sorted(e, key=edge_pos.get)
        d = dict(weight=edge_pos[v] - edge_pos[u] if weight_by_time else 1)

        return u, v, d

    L = nx.DiGraph()
    L.add_edges_from(map(create_edge, H.get_linegraph().edges()))

    return L


def draw_temporal_line_graph(G, pos=None, layout=default_layout, path=[], labels={}):
    if pos is None:
        pos = layout(G)

    path_edges = set(zip(path[:-1], path[1:]))

    nx.draw_networkx_nodes(G, pos)

    nx.draw_networkx_labels(G, pos, labels={v: labels.get(v, v) for v in G})

    nx.draw_networkx_edges(
        G, pos, width=[3 if e in path_edges else 1 for e in G.edges()]
    )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    )

    return pos
