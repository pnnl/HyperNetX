import matplotlib.pyplot as plt

import hypernetx as hnx
from hypernetx.drawing import draw_storyline as ds

import networkx as nx

from collections import defaultdict

def create_temporal_incidence_graph(H, edge_pos=None, edge_order=None, weight_by_time=True):
    assert (edge_pos is None) ^ (edge_order is None),\
        'Exactly one of edge_pos and edge_order must be specified'
    
    if edge_pos is None:
        edge_pos = {
            v: i
            for i, v in enumerate(edge_order)
        }

    G = nx.DiGraph()
    
    for v in H.nodes():
        edges = sorted(H.nodes[v], key=edge_pos.get)
        for e in edges:
            w = 0 if weight_by_time else 1
            
            G.add_edge((e, v), e, weight=w)
            G.add_edge(e, (e, v), weight=0)
    
        # temporal/directed edges connecting incidences
        for e1, e2 in zip(edges[:-1], edges[1:]):
            w = edge_pos[e2] - edge_pos[e1] if weight_by_time else 0
            G.add_edge((e1, v), (e2, v), weight=w)
    
    return G

def draw_temporal_incidence_graph(G, show_weights=True):
    G.graph = dict(rankdir='LR')

    # pos = nx.kamada_kawai_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')

    def is_incidence(v):
        return type(v) is tuple

    nx.draw_networkx_nodes(
        G, pos,
        node_color='white',
        edgecolors=[
            'none' if is_incidence(v) else 'black'
            for v in G
        ],
        node_size=300,
    )
    nx.draw_networkx_labels(
        G, pos,
        labels={
            v: f'({v[0]}, {v[1]})' if is_incidence(v) else v
            for v in G
        },
        bbox=dict(color='white', alpha=.25)
    )
    nx.draw_networkx_edges(
        G, pos
    )

    if show_weights:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={
                (u, v): str(d['weight']) if d['weight'] > 0 else ''
                for u, v, d in G.edges(data=True)
            },
            rotate=False
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

def temporal_shortest_path(H, source, target, **kwargs):
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

    return cost, [
        i
        for i in path
        if i not in edges
    ]

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
