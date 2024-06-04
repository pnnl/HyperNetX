"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""

import hypernetx as hnx
import threading
import random
from hypernetx.classes.hypergraph import Hypergraph



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
    [{0: {1, 2, 3}}, {1: {4, 5, 6}}]

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (1, 4, 7), 4: (2, 5, 8), 5: (3, 6, 9)})
    >>> greedy_d_approximation(hypergraph)
    [{0: {1, 2, 3}}, {1: {4, 5, 6}}, {2: {8, 9, 7}}]


    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    >>> greedy_d_approximation(hypergraph)
    [{0: {1, 2, 3}}, {3: {5, 6, 7}}]


    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
    >>> greedy_d_approximation(hypergraph)
    [{0: {1, 2, 3, 4}}, {1: {8, 5, 6, 7}}, {2: {9, 10, 11, 12}}]

    """
    matching = []
    available_vertices = set(hypergraph.nodes)
    for edge in hypergraph.edges:
        edge_vertices = hypergraph.edges[edge]
        if len(edge_vertices) < 2:
            print("Error: Hyperedge with fewer than 2 vertices detected:", edge_vertices)
            raise ValueError("Hyperedge must have at least 2 vertices")

        if all(v in available_vertices for v in edge_vertices):
            matching.append({edge: set(edge_vertices)})
            for v in edge_vertices:
                available_vertices.remove(v)

    return matching


######      Helper functions to implement algorithm 2 +3 for hypergraph matching  #####
def maximal_matching(hypergraph):
    matching = []
    matched_vertices = set()

    for edge in hypergraph.incidence_dict.values():
        if not any(vertex in matched_vertices for vertex in edge):
            matching.append(edge)
            matched_vertices.update(edge)

    return matching

def sample_edges(hypergraph, p):
    sampled_edges = [edge for edge in hypergraph.incidence_dict.values() if random.random() < p]
    return hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(sampled_edges)})

def sampling_round(S, p, s):
    E_prime = sample_edges(S, p)
    if len(E_prime.incidence_dict.values()) > s:
        return None, E_prime
    return maximal_matching(E_prime), E_prime

def parallel_iterated_sampling(hypergraph, d, s):
    M = []
    S = hypergraph
    p = s / (5 * len(S.edges) * d) if len(S.edges) > 0 else 0

    while True:
        M_prime, E_prime = sampling_round(S, p, s)
        if M_prime is None:
            return None  # Algorithm fails if sampled edges exceed memory limit

        M.extend(M_prime)
        unmatched_vertices = set(S.nodes) - set(v for edge in M_prime for v in edge)
        induced_edges = [edge for edge in S.incidence_dict.values() if all(v in unmatched_vertices for v in edge)]
        if len(induced_edges) <= s:
            M.extend(maximal_matching(hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})))
            break
        S = hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})

    return M


def iterated_sampling(hypergraph: Hypergraph, s: int) -> list:
    """
    Algorithm 3: Iterated Sampling for Hypergraph Matching
    Uses iterated sampling to find a maximal matching in a d-uniform hypergraph.

    Parameters:
    hypergraph (Hypergraph): A Hypergraph object.
    s (int): The amount of memory available for the computer.

    Returns:
    list: The edges of the graph for the approximate matching.

    Examples:
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> sorted(iterated_sampling(hypergraph, 2))
    [(1, 2, 3), (4, 5, 6)]

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (1, 4, 7), 4: (2, 5, 8), 5: (3, 6, 9)})
    >>> sorted(iterated_sampling(hypergraph, 3))
    [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    >>> sorted(iterated_sampling(hypergraph, 3))
    [(1, 2, 3), (5, 6, 7)]

    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
    >>> sorted(iterated_sampling(hypergraph, 4))
    [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    """
    # Fetch the degree 'd' from the hypergraph
    d = max(len(edge) for edge in hypergraph.incidence_dict.values())

    def process_partition(partition, results, index):
        results[index] = parallel_iterated_sampling(partition, d, s)

    num_threads = 3
    edges_list = list(hypergraph.incidence_dict.values())
    partitions = [edges_list[i::num_threads] for i in range(num_threads)]
    hypergraph_partitions = [hnx.Hypergraph({f'e{j}': edge for j, edge in enumerate(partitions[i])}) for i in range(num_threads)]
    threads = []
    results = [None] * num_threads

    for i in range(num_threads):
        thread = threading.Thread(target=process_partition, args=(hypergraph_partitions[i], results, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    combined_matching = set()
    for result in results:
        if result:
            combined_matching.update(tuple(edge) for edge in result)

    return list(combined_matching)


if __name__ == '__main__':
    import doctest
    doctest.testmod()




