import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import hypernetx as hnx
from hypernetx.drawing import draw_storyline as ds

import networkx as nx

from collections import defaultdict

NODE_COLOR = 'black'
NODE_ENDPOINT_COLOR ='red'
NODE_LINEWIDTH = 1
NODE_PATH_LINEWIDTH = 3
HYPER_EDGE_FACECOLOR = 'white'
HYPER_EDGE_EDGECOLOR = 'darkgray'

def get_edge_pos(edge_order, edge_pos):
    assert (edge_pos is None) ^ (edge_order is None),\
            'Exactly one of edge_pos and edge_order must be specified'
        
    if edge_pos is None:
        edge_pos = {
            v: i
            for i, v in enumerate(edge_order)
        }

    return edge_pos

def create_temporal_incidence_graph(H, edge_pos=None, edge_order=None, weight_by_time=True):
    edge_pos = get_edge_pos(edge_order, edge_pos)

    G = nx.DiGraph()
    
    for v in H.nodes():
        edges = sorted(H.nodes[v], key=edge_pos.get)
        for e in edges:
            w = 0 if weight_by_time else 1
            
            G.add_edge((e, v), e, weight=w)
            G.add_edge(e, (e, v), weight=w)
    
        # temporal/directed edges connecting incidences
        for e1, e2 in zip(edges[:-1], edges[1:]):
            w = edge_pos[e2] - edge_pos[e1] if weight_by_time else 0
            G.add_edge((e1, v), (e2, v), weight=w)
    
    return G

def default_layout(G):
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
    incidence_alpha=.85,
    sm_font=8,
    md_font=10,
    lg_font=12,
    hyper_edge_node_size=300,
    incidence_node_size=500,
    ax=None
):
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
        incidence_node_size if is_incidence(v) else hyper_edge_node_size
        for v in G
    ]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=[
            hyper_edge_color if v in path_hyperedges else 'none'
            for v in G
        ],
        edgecolors=[
            'none' if is_incidence(v) else hyper_edge_color
            for v in G
        ],
        node_size=sizes,
        ax=ax
    )

    # edges
    nx.draw_networkx_labels(
        G, pos,
        font_color=hyper_edge_font_color,
        # font_weight='bold',
        font_size=lg_font,
        labels={
            v: v
            for v in G
            if not is_incidence(v)
        },
        ax=ax
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
            fc, ec  = get_incidence_colors(v)
            ax.annotate(
                ', '.join(map(str, v)), pos[v],
                va='center', ha='center',
                fontsize=md_font,
                color=ec,
                bbox=dict(
                    facecolor=(*to_rgb(fc), incidence_alpha),
                    edgecolor=ec
                )
            )

    nx.draw_networkx_edges(
        G, pos,
        width=[
            path_width if e in path_edges else width
            for e in G.edges()
        ],
        node_size=sizes,
        ax=ax
    )

    if show_weights:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={
                (u, v): str(d['weight']) if d['weight'] > 0 or show_zero_weight else ''
                for u, v, d in G.edges(data=True)
            },
            font_color=weight_font_color,
            font_weight='bold',
            font_size=sm_font,
            rotate=False,
            ax=ax
        )

def multi_source_target_dijkstra(G, sources, targets, **kwargs):
    return min(
        [    
            nx.multi_source_dijkstra(
                G,
                sources=sources,
                target=t,
                **kwargs
            )
            for t in targets
        ],
        key=lambda x: x[0]
    )

def find_incidences(H, x):
    if type(x) is tuple:
        return [x]
    elif x in H.edges:
        return [(x, v) for v in H.edges[x]]
    elif x in H.nodes:
        return [(e, x) for e in H.nodes[x]]
    return []

def temporal_shortest_path(H, source, target, return_graph=False, **kwargs):
    G = create_temporal_incidence_graph(H, **kwargs)

    cost, path = multi_source_target_dijkstra(
        G,
        sources=find_incidences(H, source),
        targets=find_incidences(H, target),
        weight='weight'
    )

 
    # removes the hyper edges from the path so it is just a sequence of incidences

    # Workaround for issue with `HypergraphView.__contains__` in `hyp_view.py` throwing
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all() 
    edges = set(H.edges) 
    path = [
        i
        for i in path
        if i not in edges
    ]

    if return_graph:
        return cost, path, G
    
    return cost, path

def encode_path(
    path,
    node_color='black',
    node_endpoint_color='red',
    node_linewidth=1,
    node_path_linewidth=3,
    edge_facecolor='white',
    edge_edgecolor='darkgray'
):
    endpoints = {path[0], path[-1]}
    segments = set(zip(path[:-1], path[1:]))
    edges = set([
        e1
        for (e1, _), (e2, _) in zip(path[:-1], path[1:])
        if e1 == e2
    ])
    
    return dict(
        edges_kwargs=dict(
            edgecolor=edge_edgecolor
        ),
        incidences_kwargs=dict(
            facecolor=defaultdict(
                lambda: edge_facecolor, 
                {
                    i: node_endpoint_color if i in endpoints and node_endpoint_color is not None else node_color
                    for i in path
                }
            ),
        ),
        segments_kwargs=dict(
            linewidths=lambda seg: node_path_linewidth if seg in segments else node_linewidth
        ),
        fill_edges=defaultdict(
            lambda: False,
            {
                i: True
                for i in edges
            }
        )
    )

def create_temporal_line_graph(H, edge_order=None, edge_pos=None, weight_by_time=True):
    edge_pos = get_edge_pos(edge_order, edge_pos)

    def create_edge(e):
        u, v = sorted(e, key=edge_pos.get)
        d = dict(
            weight=edge_pos[v] - edge_pos[u] if weight_by_time else 1
        )
        
        return u, v, d

    L = nx.DiGraph()
    L.add_edges_from(map(create_edge, H.get_linegraph().edges()))

    return L

def draw_temporal_line_graph(G, pos=None, layout=default_layout, path=[], labels={}):
    if pos is None:
        pos = layout(G)

    path_edges = set(zip(path[:-1], path[1:]))
    
    nx.draw_networkx_nodes(G, pos)

    nx.draw_networkx_labels(
        G, pos,
        labels={
            v: labels.get(v, v)
            for v in G
        }
    )

    nx.draw_networkx_edges(
        G, pos,
        width=[
            3 if e in path_edges else 1
            for e in G.edges()
        ]
    )
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={
            (u, v): d['weight']
            for u, v, d in G.edges(data=True)
        }
    )

    return pos
