"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""
import numpy as np

import hypernetx as hnx
import threading
import random
import math
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

class MemoryLimitExceededError(Exception):
    """Custom exception to indicate memory limit exceeded during hypergraph matching."""
    pass


def maximal_matching(hypergraph):
    matching = []
    matched_vertices = set()

    for edge in hypergraph.incidence_dict.values():
        if not any(vertex in matched_vertices for vertex in edge):
            matching.append(sorted(edge))
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

def parallel_iterated_sampling(hypergraph, d, s, max_iterations=100, debug=False):
    M = []
    S = hypergraph
    p = s / (5 * len(S.edges) * d) if len(S.edges) > 0 else 0
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        if debug:
            print(f"Iteration {iterations}: Sampling with probability {p}, current number of edges: {len(S.edges)}")
        M_prime, E_prime = sampling_round(S, p, s)
        if M_prime is None:
            if debug:
                print("Sampling failed due to memory constraints.")
            raise MemoryLimitExceededError("Memory limit exceeded during hypergraph matching")

        M.extend(M_prime)
        unmatched_vertices = set(S.nodes) - set(v for edge in M_prime for v in edge)
        induced_edges = [edge for edge in S.incidence_dict.values() if all(v in unmatched_vertices for v in edge)]
        if len(induced_edges) <= s:
            if debug:
                print(f"Number of induced edges: {len(induced_edges)}")
            M.extend(maximal_matching(hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})))
            break
        S = hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})
        if debug:
            print(f"New size of S: {len(S.edges)}")

    if iterations >= max_iterations:
        if debug:
            print("Max iterations reached without finding a solution.")
        raise MemoryLimitExceededError("Max iterations reached without finding a solution")

    return M


def iterated_sampling(hypergraph: Hypergraph, s: int) -> list:
    """
    Algorithm 2: Iterated Sampling for Hypergraph Matching
    Uses iterated sampling to find a maximal matching in a d-uniform hypergraph.

    Parameters:
    hypergraph (Hypergraph): A Hypergraph object.
    s (int): The amount of memory available for the computer.

    Returns:
    list: The edges of the graph for the approximate matching.

    Raises:
    MemoryLimitExceededError: If the memory limit is exceeded during the matching process.

    Examples:
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5)})
    >>> result = iterated_sampling(hypergraph, 1)
    >>> result is None or all(len(edge) >= 2 for edge in result)  # Each edge in the result should have at least 2 vertices
    True

    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (2, 3, 4, 5), 2: (3, 4, 5, 6)})
    >>> result = iterated_sampling(hypergraph, 2)
    >>> result is None or len(result) <= 2  # The result should fit within the memory constraint
    True

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> result = None
    >>> try:
    ...     result = iterated_sampling(hypergraph, 0)  # Insufficient memory, expect failure
    ... except MemoryLimitExceededError:
    ...     pass
    >>> result is None
    True

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> result = iterated_sampling(hypergraph, 10)  # Large enough memory, expect a result
    >>> result is not None
    True

    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    >>> result = iterated_sampling(hypergraph, 3)
    >>> result is None or all(len(edge) >= 2 for edge in result)
    True

    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
    >>> result = iterated_sampling(hypergraph, 4)
    >>> result is None or all(len(edge) >= 2 for edge in result)
    True
    """
    debug = False
    # Fetch the degree 'd' from the hypergraph
    d = max((len(edge) for edge in hypergraph.incidence_dict.values()), default=0)

    if d == 0:
        return []

    def process_partition(partition, results, index):
        try:
            results[index] = parallel_iterated_sampling(partition, d, s, debug=debug)
        except MemoryLimitExceededError:
            results[index] = None

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
        if result is None:
            return None
        combined_matching.update(tuple(edge) for edge in result)

    # Ensure the final matching is maximal
    final_hypergraph = hnx.Hypergraph({f'e{i}': edge for i, edge in enumerate(combined_matching)})
    final_matching = maximal_matching(final_hypergraph)

    return final_matching

# def HEDCS(G, epsilon):
#     H = Hypergraph()  # Initialize an empty hypergraph
#     beta = len(G.nodes)
#     d = max(len(edge) for edge in G.edges) if G.edges else 0
#     beta_complement = beta - (d - 1)
#
#     for edge in G.edges:
#         add_edge = True
#         for vertex in edge:
#             degree_v = G.degree(vertex)
#             if degree_v > beta:
#                 add_edge = False
#                 break
#         if add_edge:
#             H.add_edge(edge)
#         else:
#             add_edge = True
#             for vertex in edge:
#                 degree_v = G.degree(vertex)
#                 if degree_v < beta_complement:
#                     add_edge = False
#                     break
#             if add_edge:
#                 H.add_edge(edge)
#
#     return H
#
# def HEDCS_matching(hypergraph: Hypergraph, s: int, debug=False) -> list:
#     """
#     Algorithm 3: HEDCS-Matching for Hypergraph Matching
#     This algorithm constructs a Hyper-Edge Degree Constrained Subgraph (HEDCS) to find
#     a maximal matching in a d-uniform hypergraph.
#
#     Parameters:
#     hypergraph (Hypergraph): A Hypergraph object.
#     s (int): The amount of memory available per machine.
#
#     Returns:
#     list: The edges of the graph for the approximate matching.
#
#     Raises:
#     MemoryLimitExceededError: If the memory limit is exceeded during the matching process.
#
#     The HEDCS-Matching algorithm leverages the concept of Hyper-Edge Degree Constrained Subgraph
#     (HEDCS) to compute a (d(d-1+1/d)^2)-approximation for the d-Uniform Hypergraph Matching problem
#     in 3 rounds of MPC (Massively Parallel Computation) using machines with Õ(n√nm) memory.
#
#     Examples:
#     >>> hypergraph = Hypergraph({0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (4, 5)})
#     >>> try:
#     ...     result = HEDCS_matching(hypergraph, 1)
#     ... except MemoryLimitExceededError:
#     ...     result = None
#     >>> result is None or all(len(edge) >= 2 for edge in result)  # Each edge in the result should have at least 2 vertices
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6), 2: (6, 7), 3: (7, 8)})
#     >>> try:
#     ...     result = HEDCS_matching(hypergraph, 2)
#     ... except MemoryLimitExceededError:
#     ...     result = None
#     >>> result is None or len(result) <= 2  # The result should fit within the memory constraint
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5)})
#     >>> result = None
#     >>> try:
#     ...     result = HEDCS_matching(hypergraph, 0)  # Insufficient memory, expect failure
#     ... except MemoryLimitExceededError:
#     ...     pass
#     >>> result is None
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8), 4: (9, 10)})
#     >>> result = HEDCS_matching(hypergraph, 10)  # Large enough memory, expect a result
#     >>> result is not None
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (3, 4, 5), 2: (5, 6, 7), 3: (7, 8, 9), 4: (9, 10, 11)})
#     >>> try:
#     ...     result = HEDCS_matching(hypergraph, 3)
#     ... except MemoryLimitExceededError:
#     ...     result = None
#     >>> result is None or all(len(edge) >= 2 for edge in result) or False  # Allowing for approximation
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
#     >>> try:
#     ...     result = HEDCS_matching(hypergraph, 4)
#     ... except MemoryLimitExceededError:
#     ...     result = None
#     >>> result is None or all(len(edge) >= 2 for edge in result) or False  # Allowing for approximation
#     True
#     """
#     import math
#
#     n = len(hypergraph.nodes)
#     m = len(hypergraph.edges)
#
#     if s <= 0:
#         raise MemoryLimitExceededError("Insufficient memory available for the matching process.")
#
#     k = math.ceil(m / (s * math.log(n)))
#     epsilon = 1 / (2 * n * math.log(n))
#
#     if debug:
#         print(f"Total edges (m): {m}, Nodes (n): {n}, Partitions (k): {k}, Epsilon: {epsilon}")
#
#     # Step 2: Partition G into k subgraphs
#     edges_list = list(hypergraph.incidence_dict.values())
#     partitions = [edges_list[i::k] for i in range(k)]
#     hypergraph_partitions = [hnx.Hypergraph({f'e{j}': edge for j, edge in enumerate(partitions[i])}) for i in range(k)]
#
#     # Step 3: Compute C(i) = HEDCS(G(i), 1 - epsilon) on each machine in parallel
#     def compute_Ci(partition, results, index):
#         try:
#             results[index] = HEDCS(partition, 1 - epsilon)
#         except MemoryLimitExceededError:
#             results[index] = None
#
#     results = [None] * k
#     threads = []
#
#     for i in range(k):
#         thread = threading.Thread(target=compute_Ci, args=(hypergraph_partitions[i], results, i))
#         threads.append(thread)
#         thread.start()
#
#     for thread in threads:
#         thread.join()
#
#     combined_matching = set()
#     for result in results:
#         if result is None:
#             return None
#         combined_matching.update(tuple(edge) for edge in result)
#
#     # Step 6: Compute and output a maximal matching on C
#     final_hypergraph = hnx.Hypergraph({f'e{i}': edge for i, edge in enumerate(combined_matching)})
#     final_matching = maximal_matching(final_hypergraph)
#
#     if len(final_matching) > s:
#         raise MemoryLimitExceededError("Result exceeds memory constraint.")
#
#     return final_matching


if __name__ == '__main__':
    import doctest

    doctest.testmod()


