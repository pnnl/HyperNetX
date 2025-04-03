import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.collections import LineCollection, PolyCollection

from collections import defaultdict

def get_storyline_graph(H, x):
    G = nx.DiGraph()
    
    for v in H.nodes():
        sorted_edges = sorted(H.nodes[v], key=x.get)
    
        for ei, ej in zip(sorted_edges[:-1], sorted_edges[1:]):
            if G.has_edge(ei, ej):
                G.get_edge_data(ei, ej)['lines'].append(v)
            else:
                G.add_edge(ei, ej, lines=[v])

    return G

def get_parent_graph(G, should_merge):
    
    Gc = nx.DiGraph()
    
    Gc.add_nodes_from(G)
    Gc.add_edges_from([
        (u, v)
        for u, v in G.edges()
        if should_merge(u, v)
    ])
    
    parents = {}
    Gp = nx.DiGraph()

    for i, ci in enumerate(nx.weakly_connected_components(Gc)):
        Gp.add_node(i, children=ci)
        for v in ci:
            parents[v] = i

    for u, v, d in G.edges(data=True):
        pu, pv = map(parents.get, (u, v))
        if pu != pv:
            Gp.add_edge(pu, pv, **d)

    return Gp, parents

def setup_bounds(G, x, radius=.25, dummy_threshold=4):
    # Create a copy of the input graph, because the structure may change due to dummy nodes
    Gd = nx.DiGraph()
    
    for v, d in G.nodes(data=True):
        pos = list(map(x.get, d['children']))
        Gd.add_node(v, start=min(pos) - radius, end=max(pos) + radius, **d)

    for u, v, d in G.edges(data=True):
        ud = Gd.nodes[u]
        vd = Gd.nodes[v]
        
        ue = ud['end']
        vs = vd['start']
        
        if vs - ue <= dummy_threshold:
            Gd.add_edge(u, v, **d)
        else:
            w = f'{u}__{v}'
            Gd.add_node(w, start=ue + 1, end=vs - 1, lines=d['lines'], children=[], is_dummy=True)
            Gd.add_edge(u, w)
            Gd.add_edge(w, v)

    return Gd

def setup_lines(H, G):
    for v, d in G.nodes(data=True):
        # gather all the storylines going through the parent node, sorted by key    
        d['lines'] = list(set(v for e in d['children'] for v in H.edges[e]))

def sort_storylines(Gp, key):
    for _, d in Gp.nodes(data=True):
        d['lines'] = sorted(d['lines'], key=key)
        
def storyline_layout_graphviz(G):
    G.graph['rankdir'] = 'LR'
    return nx.nx_agraph.graphviz_layout(G, prog='dot')

class Storyline:
    def __init__(self, H, edge_order=None, node_order=None):
        self.H = H

        combined_order = nx.spectral_ordering(H.bipartite())
        
        def create_order(entity_set, override):
            if override is None:
                return [v for v in combined_order if v in entity_set]
            return override
        
        self.node_order = create_order(self.H.nodes, node_order)
        self.edge_order = create_order(self.H.edges, edge_order)
        
        # mapping from edges to x-coordinate
        self.x = {
            e: i
            for i, e in enumerate(self.edge_order)
        }
      
        self.G = get_storyline_graph(self.H, self.x)

        def merge_equivalent_left(u, v):
            return self.G.out_degree(v) <= 1 and self.G.in_degree(v) <= 1 and\
                set(self.H.edges[u]) == set(self.H.edges[v])
            
        Gp, self.parents = get_parent_graph(self.G, merge_equivalent_left)
        setup_lines(self.H, Gp)
        
        self.Gp = setup_bounds(
            Gp,
            self.x
        )

        sort_storylines(self.Gp, ({v: i for i, v in enumerate(self.node_order)}).get)


    def validate_storyline_graph_direction(self):
        return {
            (u, v)
            for u, v in self.G.edges()
            if self.x[u] > self.x[v]
        }

    def validate_storyline_parent_graph_grouping(self):
        def test_all_same(children):
            nodes = [
                set(self.H.edges[c])
                for c in children
            ]

            for d in nodes:
                if d != nodes[0]:
                    return False

            return True
                
            
        return {
            v
            for v, d in self.Gp.nodes(data=True)
            if not test_all_same(d['children'])
        }
    
    def layout_graphviz(self, return_pos=False):
        self.Gp.graph['rankdir'] = 'LR'
        pos = nx.nx_agraph.graphviz_layout(self.Gp, prog='dot')

        if return_pos:
            return pos
        
        return {
            k: yk
            for k, (_, yk) in pos.items()
        }
    
    def validate_parent_graph_direction(self):
        return {
            (u, v)
            for u, v in self.Gp.edges()
            if self.Gp.nodes[u]['end'] > self.Gp.nodes[v]['start']
        }
            

    def get_line_coordinates(self, y):
        lines = defaultdict(list)

        for e, d in self.Gp.nodes(data=True):
            for dy, v in enumerate(d['lines']):
                lines[v].append((d['start'], d['end'], y[e], dy))

        return [
            sorted(lines[v], key=lambda d: d[0])
            for v in self.H.nodes()
        ]

    def get_storylines(self, y, y_spacing=1, **kwargs):

        def interpolate(points):
            smooth = PchipInterpolator(*points.T)
            points_interp = []

            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
                if y1 == y2:
                    points_interp.append(((x1, y1), (x2, y2)))
                else:
                    X = np.linspace(x1, x2, 50)[1:-1]
                    Y = smooth(X)

                    points_interp.append(np.vstack((X, Y)).T)

            return np.vstack(points_interp)
        
        return LineCollection(
            [
                interpolate(np.vstack([
                    np.array([
                        (x1, y + y_spacing*dy),
                        (x2, y + y_spacing*dy)
                    ])
                    for x1, x2, y, dy in c
                ]))
                for c in self.get_line_coordinates(y)
            ],
            **kwargs
        )

    def get_parent_graph_nodes(self, y, y_spacing=1, **kwargs):
        return PolyCollection(
            [
                [
                    (d['start'], y[v] - y_spacing/2),
                    (d['start'], y[v] + y_spacing*(len(d['lines']) - .5)),
                    (d['end'], y[v] + y_spacing*(len(d['lines']) - .5)),
                    (d['end'], y[v] - y_spacing/2)
                ]
                for v, d in self.Gp.nodes(data=True)
            ],
            **kwargs
        )
        
    def get_parent_graph_links(self, y, y_spacing=1, **kwargs):
        def midpoint(v):
            return y[v] + y_spacing*(len(self.Gp.nodes[v]['lines']) - 1)/2
        
        return LineCollection(
            [
                [
                    (self.Gp.nodes[u]['end'], midpoint(u)),
                    (self.Gp.nodes[v]['start'], midpoint(v)),
                ]
                for u, v in self.Gp.edges()
            ],
            **kwargs
        )
        
    def suggest_size(self, inches_per_edge=.5, inches_per_node=.25):
        return (inches_per_edge*len(self.H.edges), inches_per_node*len(self.H.nodes))

