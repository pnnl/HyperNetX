import hypernetx as hnx
from hypernetx.drawing.util import (
    inflate_kwargs,
    transpose_inflated_kwargs,
    inflate,
    get_frozenset_label,
)

from hypernetx.drawing.rubber_band import add_edge_defaults

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.collections import LineCollection, PolyCollection, EllipseCollection

from collections import defaultdict

EDGE_WIDTH = .5

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
        
class Storyline:
    def __init__(self, H, edge_order=None, node_order=None, y_spacing=1):
        self.H = H
        self.y_spacing = y_spacing

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
            
        def merge_equivalent_by_degree(u, v):
            return self.G.out_degree(u) == 1 and self.G.in_degree(v) == 1 and\
                set(self.H.edges[u]) == set(self.H.edges[v])
            
        Gp, self.parents = get_parent_graph(self.G, merge_equivalent_by_degree)
        setup_lines(self.H, Gp)
        
        self.Gp = setup_bounds(
            Gp,
            self.x,
            radius=EDGE_WIDTH/2
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
        
        self.y = {
            k: yk
            for k, (_, yk) in pos.items()
        }

        return self.y
    
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

    def get_storylines(self, y, **kwargs):

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

            points_interp = np.vstack(points_interp)

            # adjust first and last x coordinate on storyline so it starts in the center of the edge
            points_interp[0, 0] += EDGE_WIDTH/2
            points_interp[-1, 0] -= EDGE_WIDTH/2

            return points_interp
        
        return LineCollection(
            [
                interpolate(np.vstack([
                    np.array([
                        (x1, y + self.y_spacing*dy),
                        (x2, y + self.y_spacing*dy)
                    ])
                    for x1, x2, y, dy in c
                ]))
                for c in self.get_line_coordinates(y)
            ],
            **kwargs
        )
    
    def get_edges(self, y, y_cap_scale=4, **kwargs):
        r = EDGE_WIDTH/2

        theta = np.linspace(0, np.pi, 21)
        half_circle = np.array([
            r*np.cos(theta),
            r*y_cap_scale*self.y_spacing*np.sin(theta)
        ]).T

        def make_edge(v):
            x = self.x[v]
            y1 = y[self.parents[v]]
            y2 = y1 + self.y_spacing*(len(self.H.edges[v]) - 1)

            return np.vstack([
                half_circle*np.array([1, -1]) + np.array([x, y1]),
                (half_circle + np.array([x, y2]))[::-1]
            ])
        
        return PolyCollection(map(make_edge, self.G), **kwargs)
    
    def get_incidences(self, y, ax=None, return_index=False, **kwargs):

        ax = ax or plt.gca()

        index = [
            (p, e, v, i)
            for p, d in self.Gp.nodes(data=True)
            for e in d['children']
            for i, v in enumerate(d['lines'])
        ]

        offsets = np.array([
            (self.x[e], y[p] + self.y_spacing*i)
            for (p, e, v, i) in index
        ])

        sizes = EDGE_WIDTH/3

        circles = EllipseCollection(
            widths=sizes,
            heights=sizes,
            angles=0,
            units="x",
            offsets=offsets,
            transOffset=ax.transData,
        )

        if return_index:
            return circles, index
        
        return circles

    
    # visualizations for debugging

    def get_parent_graph_nodes(self, y, **kwargs):
        return PolyCollection(
            [
                [
                    (d['start'], y[v] - self.y_spacing/2),
                    (d['start'], y[v] + self.y_spacing*(len(d['lines']) - .5)),
                    (d['end'], y[v] + self.y_spacing*(len(d['lines']) - .5)),
                    (d['end'], y[v] - self.y_spacing/2)
                ]
                for v, d in self.Gp.nodes(data=True)
            ],
            **kwargs
        )
        
    def get_parent_graph_links(self, y, **kwargs):
        def midpoint(v):
            return y[v] + self.y_spacing*(len(self.Gp.nodes[v]['lines']) - 1)/2
        
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
    
def suggest_size(H, inches_per_edge=.5, inches_per_node=.25):
    return (inches_per_edge*len(H.edges), inches_per_node*len(H.nodes))

def draw_storyline(
    H,
    ax=None,
    y_spacing=1,
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
    y_cap_scale=4
):
    ax = ax or plt.gca()

    edges_kwargs = add_edge_defaults(H, edges_kwargs)

    default_node_color = "black"
    
    self = Storyline(
        H,
        edge_order=edge_order,
        node_order=node_order,
        y_spacing=y_spacing
    )
    
    y = self.layout_graphviz()
        
    edges = self.get_edges(
        y, y_cap_scale=y_cap_scale,
        **inflate_kwargs(H, edges_kwargs)
    )

    if fill_edges:
        color = edges.get_edgecolors() + np.array([0, 0, 0, fill_edge_alpha])
        edges.set_facecolors(color)
    
    storylines = self.get_storylines(
        y,
        **inflate_kwargs(H, {'edgecolors': default_node_color, **nodes_kwargs})
    )

    incidences, index = self.get_incidences(
        y,
        ax = ax,
        return_index=True
    )
    
    node_edgecolor_dict = dict(zip(self.H.nodes, storylines.get_edgecolors()))
    
    incidences.set_edgecolors([
        node_edgecolor_dict[v]
        for p, e, v, i in index
    ])

    # todo: facecolors could be specified in nodes_kwargs
    incidences.set_facecolors(incidences.get_edgecolors())

    for c in (edges, storylines, incidences):
        ax.add_collection(c)

    ax.autoscale_view()

    return self
