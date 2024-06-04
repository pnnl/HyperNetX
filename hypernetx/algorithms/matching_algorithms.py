"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""
import threading

from hypernetx.classes.hypergraph import Hypergraph

#
def hedcs(hypergraph: Hypergraph, beta: int, beta_minus: int) -> Hypergraph:
    """
    Constructs a HEDCS subgraph from a given hypergraph.
    """
    edges_to_add = {}
    degree_counter = {v: 0 for v in hypergraph.nodes}

    def process_edge(edge, edge_vertices):
        add_edge = True
        print(f"Checking edge {edge} with vertices {edge_vertices}")
        for v in edge_vertices:
            degree_v = degree_counter[v] + 1
            print(f"Vertex {v} has degree {degree_v} in the subgraph")
            if degree_v > beta:
                print(f"Vertex {v} exceeds the degree constraint of {beta}")
                add_edge = False
                break
        if add_edge:
            print(f"Adding edge {edge} to HEDCS subgraph")
            edges_to_add[edge] = edge_vertices
            for v in edge_vertices:
                degree_counter[v] += 1

    threads = []
    for edge in hypergraph.edges:
        edge_vertices = hypergraph.edges[edge]
        thread = threading.Thread(target=process_edge, args=(edge, edge_vertices))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    limited_edges_to_add = dict(list(edges_to_add.items())[:beta])
    H = Hypergraph(limited_edges_to_add)
    print(f"Final HEDCS subgraph edges: {list(H.edges)}")
    return H


def greedy_d_approximation(hypergraph: Hypergraph) -> list:
    """
    Algorithm 1: Greedy d-Approximation for Hypergraph Matching
    Finds a greedy d-approximation for hypergraph matching.

    Parameters:
    hypergraph (Hypergraph): A Hypergraph object.

    Returns:
    list: the edges of the graph for the approximate matching.

    Examples:
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> greedy_d_approximation(hypergraph)
    [(1, 2, 3), (4, 5, 6)]

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (1, 4, 7), 4: (2, 5, 8), 5: (3, 6, 9)})
    >>> greedy_d_approximation(hypergraph)
    [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    >>> greedy_d_approximation(hypergraph)
    [(1, 2, 3), (5, 6, 7)]


    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
    >>> greedy_d_approximation(hypergraph)
    [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 1)]

    """
    matching = []
    available_vertices = set(hypergraph.nodes)
    print("Available vertices at start:", available_vertices)

    for edge in hypergraph.edges:
        print("Processing edge:", edge)
        edge_vertices = hypergraph.edges[edge]
        if len(edge_vertices) < 2:
            print("Error: Hyperedge with fewer than 2 vertices detected:", edge_vertices)
            raise ValueError("Hyperedge must have at least 2 vertices")

        if all(v in available_vertices for v in edge_vertices):
            print("Adding edge to matching:", edge_vertices)
            matching.append({edge: set(edge_vertices)})
            for v in edge_vertices:
                available_vertices.remove(v)
                print(f"Removing vertex {v} from available vertices:", available_vertices)

    return matching

# def hedcs_based_approximation(hypergraph: Hypergraph, d: int, s: int) -> list:
#     """
#     Algorithm 2: HEDCS-Based Approximation for Hypergraph Matching
#     Computes an approximation to the maximum matching in hypergraphs using HyperEdge Degree Constrained Subgraph (HEDCS).
#
#     Parameters:
#     hypergraph (Hypergraph): A Hypergraph object.
#     d (int): The uniform size of each hyperedge in the hypergraph.
#     s (int): The amount of memory available for the computer.
#
#     Returns:
#     list: The edges of the graph for the approximate matching.
#
#     Examples:
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
#     >>> hedcs_based_approximation(hypergraph, 3, 1)
#     [(1, 2, 3), (4, 5, 6)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (1, 4, 7), 4: (2, 5, 8), 5: (3, 6, 9)})
#     >>> hedcs_based_approximation(hypergraph, 3, 2)
#     [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
#     >>> hedcs_based_approximation(hypergraph, 3, 2)
#     [(1, 2, 3), (5, 6, 7)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
#     >>> hedcs_based_approximation(hypergraph, 4, 3)
#     [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
#     """
#     if d < 2:
#         raise ValueError("d must be at least 2")
#
#     beta = d  # Assuming uniform hyperedges of size d
#     beta_minus = 1
#     subgraph = hedcs(hypergraph, beta, beta_minus)
#
#     matching = list(subgraph.edges)
#     print(f"Matching edges: {matching}")
#     return matching

# def iterated_sampling(hypergraph: Hypergraph, d: int ,s: int) -> list:
#     """
#     Algorithm 3: Iterated Sampling for Hypergraph Matching
#     Uses iterated sampling to find a maximal matching in a d-uniform hypergraph.
#
#     Parameters:
#     hypergraph (Hypergraph): A Hypergraph object.
#     d (int): The uniform size of each hyperedge in the hypergraph.
#     s (int): The amount of memory available for the computer
#
#     Returns:
#     list: The edges of the graph for the approximate matching.
#
#     Examples:
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
#     >>> iterated_sampling(hypergraph, 3, 10)
#     [(1, 2, 3), (4, 5, 6)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (1, 4, 7), 4: (2, 5, 8), 5: (3, 6, 9)})
#     >>> iterated_sampling(hypergraph, 3, 10)
#     [(1, 2, 3), (4, 5, 6), (7, 8, 9)] or [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
#     >>> iterated_sampling(hypergraph, 3, 10)
#     [(1, 2, 3), (5, 6, 7)] or [(2, 3, 4), (5, 6, 7)]
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
#     >>> iterated_sampling(hypergraph, 4, 10)
#     [(1, 2, 3, 4), (5, 6, 7, 8)] or [(9, 10, 11, 12), (13, 14, 15, 1)]
#     """
#     return []

if __name__ == '__main__':
    import doctest
    doctest.testmod()

