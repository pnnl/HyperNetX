import networkx as nx

class NetworkSimplex:
    def __init__(self):
        pass

    def validate_layers(self, G, L):
        """ Ensures that the layering is valid.

        A layering is valid if for each directed edge (u,v) in E,  L(u) > L(v).

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph
            
        L : dict
            mapping of each vertex in G to an integer level

        """
        for u,v in G.edges():
            assert L[u] > L[v], "(%s,%s) is not ordered: %d -> %d"%(repr(u),repr(v), L[u], L[v])

    def longest_path_levels(self, G):
        """ Computes a layering of the directed graph G

        A layering is a mapping of vertices in G to integer levels L. The layering 
        must be consistent--that is every source node in G is above every target.
        This algorithm uses longest path layering, which is essentially a direct
        implementation of this constraint, using recursion and memoization to find
        the result in O(|E|) time.

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph
            
        Returns
        -------
        L : mapping of the vertices in G to integer levels

        Notes
        -----
        By default, every sink vertex is assigned to level 0 by default, which may 
        result in a poor quality layering.  Thus, this algorithm is used as a first
        pass, and improved using other methods.


        References
        ----------
        .. [1] Graph Drawing Handbook p. 420
        .. [2] Gansner / A Technique for Drawing Directed Graphs
        """

        L = {}
        
        def get_level(u):
            if u in L:
                return L[u]
                
            if G.out_degree(u) == 0:
                lu = 0
            else:
                lu = 1 + max(
                    get_level(v)
                    for u,v in G.out_edges(u)
                )

            L[u] = lu
            return lu
        
        for v in G:
            get_level(v)
            
        return L

    def feasible_tree(self, G):
        """ Generates a feasible tree given the directed graph G

        A feasible tree is a spanning tree whose traversal (in any order) produces
        a valid layering.  Not all spanning trees are feasible.  The tree is found
        by generating an initial layering of G using longest path layering and then
        finding a minimum spanning tree, where the total slack is minimized.

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph
            
        Returns
        -------
        T : an undirected spanning tree of G

        Notes
        -----
        The weight of each directed edge (u,v) in the graph is set to the slack of
        the edge, given a layering L, which is defined as:

            slack(u,v) = L[u] - L[v]

        References
        ----------
        .. [1] Graph Drawing Handbook p. 420
        .. [2] Gansner / A Technique for Drawing Directed Graphs
        """

        L = self.longest_path_levels(G)
        self.validate_layers(G,L)
        
        # compute a feasbile tree
        for u,v,d in G.edges(data=True):
            # compute the slack and save it as an edge property
            d['slack'] = L[u] - L[v]

        return nx.minimum_spanning_tree(G.to_undirected(), weight='slack')

    def feasible_tree_to_levels(self, G,T):
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

        References
        ----------
        .. [1] Graph Drawing Handbook p. 420
        .. [2] Gansner / A Technique for Drawing Directed Graphs
        """

        # traverse the tree and propagate the level
        L = {}
        for u,v in nx.dfs_edges(T):
            lu = L.setdefault(u, 0)
            L[v] = lu + (1,-1)[G.has_edge(u,v)]
        
        self.validate_layers(G,L)
        return L

    def __call__(self, G, weight='weight', max_iter=1000):
        """Network simplex algorithm to compute a nice DiGraph layering

        First computes a feasible tree using longest path layering.  Then
        iteratively changes the tree by removing & adding edges until the tree
        is optimal.  The inital feasible tree is usually very close to optimal
        so the algorithm performs few iterations.

        Parameters
        ----------
        G : DiGraph
            directed acyclic graph to compute the layering on

        weight : string
            name of weight property on edges

        max_iter : int
            just in case of cycling, there is a cap on the maximum number of iterations
            
        Returns
        -------
        levels : dict
            a mapping of nodes in the aggragate graph G to heights

        Notes
        -----
        Currently using the O(|V|*|E|*n_iter) method--easier to implement.

        References
        ----------
        .. [1] Gansner / A Technique for Drawing Directed Graphs
        """

        def get_cut_value(u,v,G,T,L):
            # fix direction of edge
            if not G.has_edge(u,v):
                u,v = v,u
                assert G.has_edge(u,v)
                
            tree.remove_edge(u,v)
            # determine the head and tail components (also works if the tree is really a forest)
            for ci in map(set, nx.connected_components(tree)):
                if u in ci:
                    tail = ci
                if v in ci:
                    head = ci
            tree.add_edge(u,v)
            
            cut_value = 0
            slack = {}
            
            for i,j,d in G.edges(data=True):
                s = 0
                # head to tail
                if i in head and j in tail:
                    s = -1
                    slack[i,j] = L[i] - L[j]
                # tail to head
                if i in tail and j in head:
                    s = 1
                cut_value += s*d.get(weight, 1.0)
            
            # if the cut value is negative and there is a replacement
            if cut_value < 0 and slack:
                return min(slack, key=slack.get)
        
        # get an initial feasible tree 
        tree = self.feasible_tree(G)

        # get the levels of this tree (used to calculate slack)
        levels = self.feasible_tree_to_levels(G, tree)
        
        # cycle through the edges in the tree until no changes are made
        n_iter = 0
        n_cuts = 1
        while n_cuts and n_iter < max_iter:
            n_cuts = 0

            # for each edge in the tree
            for u,v in tree.edges():

                assert tree.has_edge(u,v)

                # check if the edge should be cut
                e = get_cut_value(u,v,G,tree,levels)
                if e and e != (u,v):

                    # if so, cut the edge and add in the new one
                    tree.remove_edge(u,v)
                    tree.add_edge(*e)
                    
                    # recalculate the levels
                    levels = self.feasible_tree_to_levels(G, tree)
                    
                    # count the number of cuts made this round
                    n_cuts += 1

            # count the number of iterations (over tree edges)
            n_iter += 1
        
        if n_iter == max_iter:
            print( "Maximum iterations reached!")

        return levels    