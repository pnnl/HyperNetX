import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from heapdict import heapdict

import networkx as nx

class LocalCrossingReducer:
    def __init__(self, order, weight=None):
        n_levels = max([x for _, x in order]) + 1
        
        self.weight = weight or (lambda x, u, v: 1)

        self.levels = [
            list()
            for _ in range(n_levels)
        ]

        self.y = {}

        for v, x in order:
            self.y[x, v] = len(self.levels[x])
            self.levels[x].append(v)

        self.left_bias = 0.1
        self.right_bias = 0.0

    def init_swaps(self):
        self.swaps = heapdict()
        
        for x, lev in enumerate(self.levels):
            for u, v in zip(lev[:-1], lev[1:]):
                self.update_swap(x, u, v)

    def draw(self, ax = None, **kwargs):
        ax = ax or plt.gca()

        self.G = nx.DiGraph()
        for i in range(len(self.levels) - 1):
            j = i + 1
            for v in self.levels[i]:
                if self.level_has_nodes(j, v):
                    self.G.add_edge(
                        (i, v),
                        (j, v)
                    )
        
        pos = {
            (i, v): (i, self.y[i, v])
            for i, lev in enumerate(self.levels)
            for v in lev
        }

        nx.draw(
            self.G, pos,
            ax=ax,
            labels={
                v: v[1]
                for v in self.G
            },
            **{
                'node_color': 'black',
                'node_size': 25,
                **kwargs,
            }
        )

        for (x, u, v), d in self.swaps.items():
            if d < 0:
                uy, vy = self.get_y(x, u, v)
                ax.annotate(d, (x, (uy + vy)/2), ha='center', va='center')

    def get_y(self, x, *args):
        return (self.y[x, v] for v in args)
    
    def swap(self, x, u, v):
        uy, vy = self.get_y(x, u, v)
        
        # swap in lists
        l = self.levels[x]
        l[uy], l[vy] = l[vy], l[uy]

        # swap in dictionary of index
        self.y[x, u], self.y[x, v] = vy, uy

        self.update_swap(x, v, u)

        # (..., t, u, v, w, ...) =>
        # (..., t, v, u, w, ...)
        # remove (t, u), (v, w) and
        # add (t, v) and (u, w)

        # handle updates within level
        if uy > 0:
            t = self.levels[x][uy - 1]
            del self.swaps[x, t, u]
            self.update_swap(x, t, v)
            
        if vy < len(self.levels[x]) - 1:
            w = self.levels[x][vy + 1]
            del self.swaps[x, v, w]
            self.update_swap(x, u, w)

        # handle updates to left and right levels
        # only need to check if u and v are adjacent in the next and previous levels
        for x2 in (x - 1, x + 1):
            if self.level_has_nodes(x2, u, v):
                uy2, vy2 = self.get_y(x2, u, v)
                dy = vy2 - uy2

                if dy == 1:
                    self.update_swap(x2, u, v)
                if dy == -1:
                    self.update_swap(x2, v, u)

    def level_has_nodes(self, x, *args):
        return np.all([(x, u) in self.y for u in args])

    def count_crossing(self, u1, v1, u2, v2):
        return int((v1 - u1)*(v2 - u2) < 0)
        
    def crossings_decreased_if_swapped(self, x, u, v):
        assert self.level_has_nodes(x, u, v), f'Level does not contain both {u} and {v}'
        
        left = None
        right = None

        uy, vy = self.get_y(x, u, v)
        
        if self.level_has_nodes(x - 1, u, v):
            left = self.count_crossing(uy, vy, *self.get_y(x - 1, u, v))

        if self.level_has_nodes(x + 1, u, v):
            right = self.count_crossing(uy, vy, *self.get_y(x + 1, u, v))

        if left is None and right is None:
            return 0
            
        if left is None:
            return -right

        if right is None:
            return -left

        return -2*(right + self.right_bias)*(left + self.left_bias)

    def assert_order(self, x, u, v):
        uy, vy = self.get_y(x, u, v)
        assert vy - uy == 1, f'Swap out of order. y({u}) = {uy}; y({v}) = {vy}'
        
    def update_swap(self, x, u, v):
        self.assert_order(x, u, v)
        self.swaps[x, u, v] = self.weight(x, u, v)*self.crossings_decreased_if_swapped(x, u, v)

    def __call__(self, max_iters=None):
        self.num_iters = 0

        self.moves = [[], []]

        for i in range(2):
            self.init_swaps()

            while self.swaps.peekitem()[1] < 0:
                k, v = self.swaps.popitem()
                self.moves[i].append(v)

                self.swap(*k)

                self.num_iters += 1
                if max_iters is not None and self.num_iters >= max_iters:
                    break

            self.left_bias, self.right_bias = self.right_bias, self.left_bias

        return [
            OrderedDict([
                (v, i)
                for i, v in enumerate(lev)
            ])
            for lev in self.levels
        ]
