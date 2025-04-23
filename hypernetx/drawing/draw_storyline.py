import hypernetx as hnx
from hypernetx.drawing.util import (
    inflate_kwargs,
    inflate_labels
)

from . import network_simplex as ns

from hypernetx.drawing.rubber_band import add_edge_defaults

import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.collections import LineCollection, PolyCollection, EllipseCollection

from collections import defaultdict, OrderedDict
from itertools import combinations

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
        
        return PolyCollection(map(make_edge, self.H.edges), **kwargs)
    
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
    
    def get_node_labels_xy(self, y):
        
        def get_left_coord(v):
            p = self.parents[min(self.H.nodes[v], key=self.x.get)]
            d = self.Gp.nodes[p]

            return (
                d['start'] + EDGE_WIDTH/2,
                y[p] + self.y_spacing*d['lines'].index(v)
            )
        
        return np.array(list(map(get_left_coord, self.H.nodes())))

    def get_edge_labels_xy(self, y):
        return np.array([
            (self.x[e], y[self.parents[e]])
            for e in self.H.edges()
        ])
    

class SvenStoryline:
    def __init__(self, H, edge_order=None, node_order=None, debug=False):
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
        
        self.G = self.get_storyline_graph()


        order = nx.spectral_ordering(self.G, seed=123456)
        self.y_init = {
            v: i
            for i, v in enumerate(order)
        }

        self.levels = [
            OrderedDict()
            for i in range(len(self.edge_order))
        ]


        for k in order:
            if k not in self.x:
                v, i = k
                self.levels[i][v] = len(self.levels[i])

        self.parents = get_parents(self.levels)
        self.Gp = get_parent_graph(self.levels, self.parents)

        if debug:
            plt.figure(); self.draw_initial_layout()
            plt.figure(); self.draw_parent_graph()

        self.solver = ns.NetworkSimplex(self.Gp)
        self.yp = self.solver()
        self.y = {
            v: self.yp[p]
            for v, p in self.parents.items()
        }

    def get_storyline_graph(self):

        self.line_endpoints = {}
        
        G = nx.Graph()
        
        for v in self.H.nodes:
            xs = list(map(self.x.get, self.H.nodes[v]))
            self.line_endpoints[v] = start, end = (min(xs), max(xs))
        
            G.add_node((v, start)) # just in case start == end
            
            for i in range(start, end):
                G.add_edge((v, i), (v, i + 1))
        
            for e in self.H.nodes[v]:
                G.add_edge((v, self.x[e]), e)

        return G
    
    # debug visualizations
    
    def draw_initial_layout(self):
        pos = {
            v: (self.x[v] if v in self.x else v[1], self.y_init[v])
            for v in self.G
        }
        
        nx.draw(
            self.G, pos,
            with_labels=True,
            node_color='white',
            node_size=500
        )

    def draw_parent_graph(self):
        labels={
            u: ' '.join(sorted(['-'.join(map(str, k)) for k, v in self.parents.items() if u == v]))
            for u in self.parents.values()
        }

        nx.draw(
            self.Gp,
            # pos=nx.kamada_kawai_layout(self.Gp),
            with_labels=True, labels=labels,
            node_color='white',
            node_size=500
        )

    # implementing storyline interface

    def get_storylines(self, r=.25, **kwargs):

        def get_steps(v):
            start, end = self.line_endpoints[v]
            return range(start, end + 1)

        def get_radii(v, i):
            return (
                -r*(self.line_endpoints[v][0] != i),
                r*(self.line_endpoints[v][1] != i)
            )
                
        return LineCollection([
            [
                (i + dx, self.y[v, i])
                for i in get_steps(v)
                for dx in get_radii(v, i)
            ]
            for v in self.H.nodes
        ], **kwargs)

    def get_edges(self, r=.25, y_cap_scale=1, **kwargs):
        theta = np.linspace(0, np.pi, 21)
        half_circle = np.array([
            r*np.cos(theta),
            r*y_cap_scale*np.sin(theta)
        ]).T

        def make_edge(e):
            x = self.x[e]
            ys = [
                self.y[(v, x)]
                for v in self.H.edges[e]
            ]
            
            y1 = min(ys)
            y2 = max(ys)

            return np.vstack([
                half_circle*np.array([1, -1]) + np.array([x, y1]),
                (half_circle + np.array([x, y2]))[::-1]
            ])
        
        return PolyCollection(map(make_edge, self.H.edges), **kwargs)
    
    def get_incidences(self, ax=None, r=.125, **kwargs):

        ax = ax or plt.gca()

        offsets = np.array([
            (self.x[e], self.y[v, self.x[e]])
            for e in self.H.edges
            for v in self.H.edges[e]
        ])

        sizes = 2*r

        return EllipseCollection(
            widths=sizes,
            heights=sizes,
            angles=0,
            units="x",
            offsets=offsets,
            transOffset=ax.transData,
            **kwargs
        )
    
    def get_node_labels_xy(self):
        
        def get_left_coord(v):
            start, _ = self.line_endpoints[v]

            return (
                start, self.y[v, start]
            )
        
        return np.array(list(map(get_left_coord, self.H.nodes())))

    def get_edge_labels_xy(self):
        return np.array([
            (self.x[e], min(self.y[v, self.x[e]] for v in self.H.edges[e]))
            for e in self.H.edges()
        ])

    def suggest_size(self, scale=.5):
        return (
            scale*(len(self.x) + 2),
            scale*(max(self.y.values()) - min(self.y.values()) + 3)
        )

def get_crossing_graph(levels):

    G = nx.Graph()
    
    for i in range(len(levels) - 1):
        left = levels[i]
        right = levels[i + 1]

        can_cross = [
            v
            for v in left
            if v in right
        ]

        for v in can_cross:
            G.add_node((v, i))
                
        for u, v in combinations(can_cross, 2):
            if (left[u] - left[v])*(right[u] - right[v]) < 0:
                G.add_edge((u, i), (v, i))

    return G

def get_parents(levels):
    G = nx.Graph()
    
    for i, l in enumerate(levels):
        for v in l:
            G.add_node((v, i))
            
    merges = nx.maximal_independent_set(get_crossing_graph(levels))    
    
    for v, i in merges:
        G.add_edge((v, i), (v, i + 1))

    return {
        v: i
        for i, ci in enumerate(nx.connected_components(G))
        for v in ci
    }


def get_parent_graph(levels, parents):

    G = nx.DiGraph()
    G.add_nodes_from(parents.values())

    for i, lev in enumerate(map(list, levels)):
        for u, v in zip(lev[:-1], lev[1:]):
            G.add_edge(parents[(u, i)], parents[(v, i)])
            
    return G    

def draw_storyline(
    H,
    layout=None,
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
    y_cap_scale=4
):
    ax = ax or plt.gca()

    edges_kwargs = add_edge_defaults(H, edges_kwargs)

    default_node_color = "black"

    if layout is None:
        layout = SvenStoryline(
            H,
            edge_order=edge_order,
            node_order=node_order,
        )

    edges = layout.get_edges(
        **inflate_kwargs(H.edges, edges_kwargs)
    )

    if fill_edges:
        color = edges.get_edgecolors() + np.array([0, 0, 0, fill_edge_alpha])
        edges.set_facecolors(color)
    
    storylines = layout.get_storylines(
        **inflate_kwargs(H, {
            'edgecolors': default_node_color,
            **nodes_kwargs,
            'facecolors': 'none' # storylines should never have a face color
        })
    )

    incidences = layout.get_incidences(ax=ax)
    
    node_edgecolor_dict = dict(zip(H.nodes, storylines.get_edgecolors()))
    
    incidences.set_edgecolors([
        node_edgecolor_dict[v]
        for e in H.edges()
        for v in H.edges[e]
    ])

    # todo: facecolors could be specified in nodes_kwargs
    incidences.set_facecolors(incidences.get_edgecolors())

    for c in (edges, storylines, incidences):
        ax.add_collection(c)

    if with_node_labels:
        default_text_kwargs = {
            'ha': 'right',
            'va': 'center',
            'xytext': (-3, 0),
            'textcoords': 'offset pixels'
        }
        offset_xy = np.array([-.5*EDGE_WIDTH, 0])
        node_xy = layout.get_node_labels_xy()
        node_labels_and_kwargs = inflate_labels(list(H.nodes), node_labels, node_labels_kwargs)

        for (s, kwargs), xy in zip(node_labels_and_kwargs, node_xy):
            ax.annotate(s, xy + offset_xy,  **{**default_text_kwargs, **kwargs})

    if with_edge_labels:
        default_text_kwargs = {
            'ha': 'center',
            'va': 'top',
            'xytext': (0, -2),
            'textcoords': 'offset pixels'
        }

        offset_xy = np.array([0, -y_cap_scale*EDGE_WIDTH/2])
        edge_xy = layout.get_edge_labels_xy()
        edge_labels_and_kwargs = inflate_labels(list(H.edges), edge_labels, edge_labels_kwargs)

        if edge_labels_on_axis:
            ax.xaxis.set_ticks(
                edge_xy[:, 0],
                [labels for labels, _ in edge_labels_and_kwargs],
                **edge_labels_kwargs
            )
        else:
            for (s, kwargs), xy in zip(edge_labels_and_kwargs, edge_xy):
                ax.annotate(s, xy + offset_xy,  **{**default_text_kwargs, **kwargs})
        
            ax.xaxis.set_ticks([], [])

    ax.autoscale_view()

    ax.yaxis.set_ticks([], [])

    return layout

