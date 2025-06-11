import networkx as nx

from warnings import warn

def longest_path_levels(G, length='length'):
    """ Computes a feasible layering of the directed graph G

    A layering is a mapping of vertices in G = (V,E) to integer levels L. A feasible
    layering respects:
        
        L[u] <= L[v] + length(u, v) for all (u, v) in E
        
    This algorithm uses longest path layering, which is essentially a direct
    implementation of this constraint using topological sorting in O(|E|) time, 
    which avoids a potentially deep recursive call stack.

    Parameters
    ----------
    G : DiGraph
        directed acyclic graph
        
    length : str
        key into edge dictionary indicating the minimum edge length
        
    Returns
    -------
    L : mapping of the vertices in G to integer levels

    Notes
    -----
    By default, every source vertex is assigned to level 0 by default, which may 
    result in a poor quality layering. Thus, this algorithm is used as a first
    pass, and improved using other methods.


    References
    ----------
    .. [1] Tamassia, Roberto, ed. Handbook of graph drawing and visualization. CRC press, 2013. 420-421.
    """

    L = {}
    
    # iterating over nodes in topological order guarantees that all the nodes above a
    # given node will already have been added to the dictionary
    for v in nx.topological_sort(G):
        if G.in_degree(v) == 0:
            L[v] = 0
        else:
            L[v] = max(
                L[u] + d.get(length, 1)
                for u, _, d in G.in_edges(v, data=True)
            )

    return L

class NetworkSimplex:
    def __init__(self, G, weight='weight', length='length'):
        """
        G : DiGraph
            directed acyclic graph to compute the layering on

        weight : string
            name of weight property on edges
        """

        self.G = G
        self.weight_str = weight
        self.length_str = length

        # calculate initial layering
        self.components = list(nx.connected_components(G.to_undirected()))
        self.L_longest = self.L = longest_path_levels(G, length)

        # initial feasible tree generated from initial layering
        self.T = self.feasible_tree()
        self.T_init = self.T.copy()
        self.L_init = self.L = self.induce_levels(self.T)

        self.violations_initial = self.validate_layers()

    def induce_levels(self, T):
        """ Given a DiGraph and Tree, computes a layering

        Given the undirected digraph G and spanning tree T, traverses T starting at
        an arbitrary node and assigns levels.  When edge (u,v) is traversed in the 
        undirected tree T, assuming u is the vertex already visited, then the level
        of v is either u + 1 or u - 1 depending on the direction of (u,v) in G.

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph 
            
        T : Tree
            an undirected spanning tree of G

        Returns
        -------
        L : mapping of vertices in G to levels
        """

        # traverse the tree and propagate the level
        L = {}
        for e in nx.dfs_edges(T):
            u, v = self.reorient_edge(e)
            d = self.length(u, v)
            
            if u in L and v not in L:
                L[v] = L[u] + d
            elif u not in L and v in L:
                L[u] = L[v] - d
            else:
                L[u] = 0
                L[v] = d
        
        # handle isolated nodes
        for v in self.G:
            L.setdefault(v, 0)

        return L

    def validate_layers(self):
        """ Ensures that the layering is valid.

        A layering is valid if for each directed edge (u,v) in E,  L(u) > L(v).

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph
            
        L : dict
            mapping of each vertex in G to an integer level

        """

        violations = {}

        for u, v in self.G.edges():
            suv = self.slack(u, v)
            if suv < 0:
                violations[u, v] = suv

        if len(violations):
            warn(f"Infeasible layering detected ({len(violations)} edges with slack < 0 exist). See NetworkSimplex.violations_*")

        return violations

    def reorient_edge(self, e):
        u, v = e
        if self.G.has_edge(u, v):
            return e
        return v, u
        
    def length(self, u, v):
        return self.G.get_edge_data(u, v).get(self.length_str, 1)
    
    def slack(self, u, v, L=None):
        """
        Finds the slack on the edge given the current layering

        The slack is the amount of free space between two nodes.

        Parameters
        ----------
        u : hashable
            tail node
        v : hashable
            head node
        L : dict or None
            layering (defaults to current layering, self.L)
            
        Returns
        -------
        slack : int
            the slack of the edge
        
        Notes
        -----
        Slack is defined as:

            slack(u,v) = L[v] - L[u] - length(u, v)

        where the minimum distance between nodes is d(u, v) = 1, if not otherwise specified
        """
        if L is None:
            L = self.L

        return L[v] - L[u] - self.length(u, v)

    def feasible_tree(self):
        """ Generates a feasible tree given the directed graph G

        A feasible tree is a spanning tree whose traversal (in any order) produces
        a valid layering.  Not all spanning trees are feasible.  The tree is found
        by generating an initial layering of G using longest path layering and then
        finding a minimum spanning tree, where the total slack is minimized.

        Returns
        -------
        T : nx.Graph
            an undirected spanning tree of G

        Notes
        -----
        The weight of each directed edge (u,v) in the graph is set to the slack.
        """

        self.G_mst = G = nx.Graph()
        G.add_nodes_from(self.G)
        
        # compute a feasbile tree
        for u, v, d in self.G.edges(data=True):
            # compute the slack and save it as an edge property
            G.add_edge(u, v, slack=self.slack(u, v))

        # no guarantees here that this tree is tight (produces a feasible tree)
        T = nx.Graph()
        for c in self.components:
            Gc = nx.subgraph(G, c)
            T.add_nodes_from(Gc)
            T.add_edges_from(nx.minimum_spanning_tree(Gc, weight='slack').edges())

        return T
    
    def get_head_and_tail_components(self, u, v):
        """ Returns the subgraphs of the tree containing the endpoints of the edge passed in

        Temporarily breaks the tree by removing the edge (u, v), then calculates the
        connected components. Test each component for whether they contain the head
        or the tail node.

        Parameters
        ----------
        u : hashable
            tail node
        v : hashable
            head node
        
        Returns
        -------
        head : hashable
            the component of T contianing v
        tail : hashable
            the component of T contianing u

        Notes
        -----
        This method is not optimized for repeated calls.
        """

        # split the tree into 2 components by removing the edge
        self.T.remove_edge(u, v)

        # determine the head and tail components (also works if the tree is really a forest)
        for ci in map(set, nx.connected_components(self.T)):
            if u in ci:
                tail = ci
            if v in ci:
                head = ci

        # reassemble the tree
        self.T.add_edge(u, v)

        return head, tail
        
    def enter_edge(self, leave_edge):
        """ Finds a feasible edge to replace (u, v) with

        Determines the head and tail components of (u, v), then determines whether
        there is more edge weight flowing from head to tail, versus tail to head, which
        includes (u, v). If there is more weight, returns the edge pointing from head to
        tail with the least non-negative slack.
        
        Parameters
        ----------
        u : hashable
            tail of leaving edge
        v : hashable
            head of leaving node

        Returns
        -------
        e : (hashable, hashable) or None
            entering edge

        Notes
        -----
        This method is not optimized for repeated calls.
        """
        
        head, tail = self.get_head_and_tail_components(*leave_edge)
        
        cut_value = 0
        slack = {}
        
        for i, j, d in self.G.edges(data=True):
            s = 0 # remains at 0 if edge is completely within head or tail component

            if i in head and j in tail:
                # candidate replacement edges must go between head and tail
                slack[i, j] = self.slack(i, j)
                s = -1
            elif i in tail and j in head:
                # not a candidate for replacement, but counts against replacing leave_edge
                s = 1

            cut_value += s*d.get(self.weight_str, 1.0)
        
        # if the cut value is negative and there is a replacement
        if cut_value < 0 and len(slack):
            enter_edge = min(slack, key=slack.get)
            self.cuts.append((leave_edge, enter_edge, slack[enter_edge], cut_value))
            return enter_edge
    
    def __call__(self, max_iter=1000):
        """Network simplex algorithm to compute a nice DiGraph layering

        First computes a feasible tree using longest path layering.  Then
        iteratively changes the tree by removing & adding edges until the tree
        is optimal.  The inital feasible tree is usually very close to optimal
        so the algorithm performs few iterations.

        Parameters
        ----------
        max_iter : int
            just in case of cycling, there is a cap on the maximum number of iterations
            
        Returns
        -------
        L : dict
            an optimized layering of G

        Notes
        -----
        Currently using the O(|V|*|E|*n_iter) method--easier to implement.

        References
        ----------
            [1] Gansner, Emden R., et al. "A technique for drawing directed graphs." IEEE Transactions on Software Engineering 19.3 (1993): 214-230.
        """

        self.cuts = []

        # cycle through the edges in the tree until no changes are made
        n_iter = 0

        while n_iter < max_iter:
            n_cuts = 0

            # for each edge in the tree (listed, because tree may change)
            # following recommendation in paper not to restart iteration from 
            for leave_edge in map(self.reorient_edge, list(self.T.edges())):

                # check if tree has edge, because edge may have been replaced since tree edges were previously listed. Double check?
                if self.T.has_edge(*leave_edge): 
                    # check if the edge should be cut
                    e = self.enter_edge(leave_edge)
                    if e is not None and e != leave_edge:
                        # if so, cut the edge and add in the new one
                        self.T.remove_edge(*leave_edge)
                        self.T.add_edge(*e)
                        
                        # recalculate the levels
                        self.L = self.induce_levels(self.T)
                        
                        # increment the number of cuts made this round
                        n_cuts += 1

            # count the number of iterations (over tree edges)
            n_iter += n_cuts

            # exit loop if no more cuts are possbile
            if n_cuts == 0:
                break

        if n_iter == max_iter:
            warn( "Maximum iterations reached!")

        # validate L
        self.violations_final = self.validate_layers()

        return self.L

    def draw_layering(self, L=None, T=None, x=None, labels={}, with_labels=True, negative_slack_color=('black', 'red'), in_tree_width=(1, 3)):
        G = self.G

        if L is None:
            L = self.L

        if T is None:
            T = self.T

        if x is None:
            pos = {
                v: (xv, L[v])
                for v, (xv,) in nx.spectral_layout(G, dim=1).items()
            }
        else:
            pos = {
                v: (x[v], L[v])
                for v in G
            }

        def is_negative(s):
            return int(s < 0)
        
        slack = [
            self.slack(u, v, L)
            for u, v in G.edges()
        ]

        if with_labels:
            nx.draw_networkx_labels(
                G, pos,
                labels={
                    v: f'({labels.get(v, v)})'
                    for v in G
                }
            )

        nx.draw_networkx_edges(
            G, pos,
            edge_color=[
                negative_slack_color[b]
                for b in map(is_negative, slack)
            ],
            width=[
                in_tree_width[int(T is not None and T.has_edge(u, v))]
                for u, v in G.edges()
            ]
        )

        nx.draw_networkx_edge_labels( 
            G, pos,
            edge_labels={
                e: s
                for e, s in zip(G.edges(), slack)
                if s != 0
            }
        )
        