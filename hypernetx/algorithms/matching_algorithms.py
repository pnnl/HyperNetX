"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""
import numpy as np
import hypernetx as hnx
from hypernetx.classes.hypergraph import Hypergraph
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor


def approximation_matching_checking(optimal: list, approx: list):
    for e in optimal:
        count = 0
        e_checks = set(e)
        for e_m in approx:
            e_m_checks = set(e_m)
            common_elements = e_checks.intersection(e_m_checks)
            checking = bool(common_elements)
            if checking:
                count += 1
        if count < 1:
            return False
    return True


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


class NonUniformHypergraphError(Exception):
    """Custom exception to un d-uniform exceeded during hypergraph matching."""
    pass


def maximal_matching(hypergraph: Hypergraph) -> list:
    """
    Finds a maximal matching in the given hypergraph.

    Parameters:
    hypergraph (Hypergraph): The input hypergraph.

    Returns:
    list: The edges of the maximal matching.
    """
    matching = []
    matched_vertices = set()

    for edge in hypergraph.incidence_dict.values():
        if not any(vertex in matched_vertices for vertex in edge):
            matching.append(sorted(edge))
            matched_vertices.update(edge)

    return matching


def sample_edges(hypergraph: Hypergraph, p: float) -> Hypergraph:
    """
    Samples edges from the hypergraph with probability p.

    Parameters:
    hypergraph (Hypergraph): The input hypergraph.
    p (float): The probability of sampling each edge.

    Returns:
    Hypergraph: A new hypergraph containing the sampled edges.
    """
    sampled_edges = [edge for edge in hypergraph.incidence_dict.values() if random.random() < p]
    return hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(sampled_edges)})


def sampling_round(S: Hypergraph, p: float, s: int) -> tuple:
    """
    Performs a single sampling round on the hypergraph.

    Parameters:
    S (Hypergraph): The input hypergraph.
    p (float): The probability of sampling each edge.
    s (int): The maximum number of edges to include in the matching.

    Returns:
    tuple: A tuple containing the maximal matching and the sampled hypergraph.
    """
    E_prime = sample_edges(S, p)
    if len(E_prime.incidence_dict.values()) > s:
        return None, E_prime
    return maximal_matching(E_prime), E_prime


def iterated_sampling(hypergraph: Hypergraph, s: int, max_iterations: int = 100) -> list:
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
    >>> optimal_test1 = [[1, 2, 3, 4]]
    >>> approximation_matching_checking(optimal_test1,result)  # The result should fit within the memory constraint
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
    >>> optimal_test2 = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
    >>> approximation_matching_checking(optimal_test2,result)
    True
    >>> s=10
    >>> # Test with d=4
    >>> edges_d4 = {'e1': [1, 2, 3, 4], 'e2': [2, 3, 4, 5], 'e3': [3, 4, 5, 6], 'e4': [4, 5, 6, 7]}
    >>> hypergraph_d4 = hnx.Hypergraph(edges_d4)
    >>> optimal_matching_d4 = [[1, 2, 3, 4]]
    >>> approximate_matching_d4 = iterated_sampling(hypergraph_d4, s)
    >>> approximation_matching_checking(optimal_matching_d4,approximate_matching_d4)
    True

    >>> # Test with d=5
    >>> edges_d5 = {'e1': [1, 2, 3, 4, 5], 'e2': [2, 3, 4, 5, 6], 'e3': [3, 4, 5, 6, 7]}
    >>> hypergraph_d5 = hnx.Hypergraph(edges_d5)
    >>> optimal_matching_d5 = [[1, 2, 3, 4, 5]]
    >>> approximate_matching_d5 = iterated_sampling(hypergraph_d5, s)
    >>> approximation_matching_checking(optimal_matching_d5,approximate_matching_d5)
    True

    >>> # Test with d=6
    >>> edges_d6 = {'e1': [1, 2, 3, 4, 5, 6], 'e2': [2, 3, 4, 5, 6, 7], 'e3': [3, 4, 5, 6, 7, 8]}
    >>> hypergraph_d6 = hnx.Hypergraph(edges_d6)
    >>> optimal_matching_d6 = [[1, 2, 3, 4, 5, 6]]
    >>> approximate_matching_d6 = iterated_sampling(hypergraph_d6, s)
    >>> d = 6
    >>> approximation_matching_checking(optimal_matching_d6,approximate_matching_d6)
    True
    >>> # Test with a larger hypergraph
    >>> edges_large = {f'e{i}': [i, i+1, i+2] for i in range(1, 101)}
    >>> hypergraph_large = Hypergraph(edges_large)
    >>> optimal_matching_large = [edges_large[f'e{i}'] for i in range(1, 101,3)]
    >>> approximate_matching_large = iterated_sampling(hypergraph_large, s)
    >>> approximation_matching_checking(optimal_matching_large,approximate_matching_large)
    True

    """
    d = max((len(edge) for edge in hypergraph.incidence_dict.values()), default=0)
    M = []
    S = hypergraph
    p = s / (5 * len(S.edges) * d) if len(S.edges) > 0 else 0
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        M_prime, E_prime = sampling_round(S, p, s)
        if M_prime is None:
            raise MemoryLimitExceededError("Memory limit exceeded during hypergraph matching")

        M.extend(M_prime)
        unmatched_vertices = set(S.nodes) - set(v for edge in M_prime for v in edge)
        induced_edges = [edge for edge in S.incidence_dict.values() if all(v in unmatched_vertices for v in edge)]
        if len(induced_edges) <= s:
            M.extend(maximal_matching(hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})))
            break
        S = hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})
        p = s / (5 * len(S.edges) * d) if len(S.edges) > 0 else 0

    if iterations >= max_iterations:
        raise MemoryLimitExceededError("Max iterations reached without finding a solution")
    return M


# def iterated_sampling(hypergraph: Hypergraph, s: int) -> list:
#     """
#     Algorithm 2: Iterated Sampling for Hypergraph Matching
#     Uses iterated sampling to find a maximal matching in a d-uniform hypergraph.
#
#     Parameters:
#     hypergraph (Hypergraph): A Hypergraph object.
#     s (int): The amount of memory available for the computer.
#
#     Returns:
#     list: The edges of the graph for the approximate matching.
#
#     Raises:
#     MemoryLimitExceededError: If the memory limit is exceeded during the matching process.
#
#     Examples:
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5)})
#     >>> result = iterated_sampling(hypergraph, 1)
#     >>> result is None or all(len(edge) >= 2 for edge in result)  # Each edge in the result should have at least 2 vertices
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (2, 3, 4, 5), 2: (3, 4, 5, 6)})
#     >>> result = iterated_sampling(hypergraph, 2)
#     >>> result is None or len(result) <= 2  # The result should fit within the memory constraint
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
#     >>> result = None
#     >>> try:
#     ...     result = iterated_sampling(hypergraph, 0)  # Insufficient memory, expect failure
#     ... except MemoryLimitExceededError:
#     ...     pass
#     >>> result is None
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
#     >>> result = iterated_sampling(hypergraph, 10)  # Large enough memory, expect a result
#     >>> result is not None
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
#     >>> result = iterated_sampling(hypergraph, 3)
#     >>> result is None or all(len(edge) >= 2 for edge in result)
#     True
#
#     >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
#     >>> result = iterated_sampling(hypergraph, 4)
#     >>> result is None or all(len(edge) >= 2 for edge in result)
#     True
#     >>> s=10
#     >>> # Test with d=4
#     >>> edges_d4 = {'e1': [1, 2, 3, 4], 'e2': [2, 3, 4, 5], 'e3': [3, 4, 5, 6], 'e4': [4, 5, 6, 7]}
#     >>> hypergraph_d4 = hnx.Hypergraph(edges_d4)
#     >>> optimal_matching_d4 = ['e1', 'e3']
#     >>> approximate_matching_d4 = iterated_sampling(hypergraph_d4, s)
#     >>> d = 4
#     >>> len(approximate_matching_d4) <= d * len(optimal_matching_d4)
#     True
#
#     >>> # Test with d=5
#     >>> edges_d5 = {'e1': [1, 2, 3, 4, 5], 'e2': [2, 3, 4, 5, 6], 'e3': [3, 4, 5, 6, 7]}
#     >>> hypergraph_d5 = hnx.Hypergraph(edges_d5)
#     >>> optimal_matching_d5 = ['e1']
#     >>> approximate_matching_d5 = iterated_sampling(hypergraph_d5, s)
#     >>> d = 5
#     >>> len(approximate_matching_d5) <= d * len(optimal_matching_d5)
#     True
#
#     >>> # Test with d=6
#     >>> edges_d6 = {'e1': [1, 2, 3, 4, 5, 6], 'e2': [2, 3, 4, 5, 6, 7], 'e3': [3, 4, 5, 6, 7, 8]}
#     >>> hypergraph_d6 = hnx.Hypergraph(edges_d6)
#     >>> optimal_matching_d6 = ['e1']
#     >>> approximate_matching_d6 = iterated_sampling(hypergraph_d6, s)
#     >>> d = 6
#     >>> len(approximate_matching_d6) <= d * len(optimal_matching_d6)
#     True
#     >>> # Test with a larger hypergraph
#     >>> edges_large = {f'e{i}': [i, i+1, i+2] for i in range(1, 101)}
#     >>> hypergraph_large = Hypergraph(edges_large)
#     >>> optimal_matching_large = [edges_large[f'e{i}'] for i in range(1, 101,3)]
#     >>> approximate_matching_large = iterated_sampling(hypergraph_large, s)
#     >>> len(approximate_matching_large) <= d * len(optimal_matching_large)
#     True
#
#
#
#
#
#     """
#     debug = False
#     # Fetch the degree 'd' from the hypergraph
#     d = max((len(edge) for edge in hypergraph.incidence_dict.values()), default=0)
#
#     if d == 0:
#         return []
#
#     def process_partition(partition, results, index):
#         try:
#             results[index] = parallel_iterated_sampling(partition, d, s, debug=debug)
#         except MemoryLimitExceededError:
#             results[index] = None
#
#     num_threads = 3
#     edges_list = list(hypergraph.incidence_dict.values())
#     partitions = [edges_list[i::num_threads] for i in range(num_threads)]
#     hypergraph_partitions = [hnx.Hypergraph({f'e{j}': edge for j, edge in enumerate(partitions[i])}) for i in range(num_threads)]
#     threads = []
#     results = [None] * num_threads
#
#     for i in range(num_threads):
#         thread = threading.Thread(target=process_partition, args=(hypergraph_partitions[i], results, i))
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
#     # Ensure the final matching is maximal
#     final_hypergraph = hnx.Hypergraph({f'e{i}': edge for i, edge in enumerate(combined_matching)})
#     final_matching = maximal_matching(final_hypergraph)
#
#     return final_matching
#


def HEDCS(G: Hypergraph, beta: int, beta_complement: int) -> Hypergraph:
    """
    Constructs a Hyper-Edge Degree Constrained Subgraph (HEDCS) from the given hypergraph G.

    Parameters:
    G (Hypergraph): The input hypergraph.
    beta (int): Degree threshold for adding edges.
    beta_complement (int): Complementary degree threshold for adding edges.

    Returns:
    Hypergraph: The constructed HEDCS.
    """
    H = Hypergraph()  # Initialize an empty hypergraph

    for edge_id in G.edges:
        edge = G.edges[edge_id]
        add_edge = True
        # Check if all vertices in the edge have a degree less than or equal to beta
        degree_sum = sum(len(G.nodes.memberships[vertex]) for vertex in edge)
        if degree_sum > beta:
            add_edge = False
        elif degree_sum < beta_complement:
            add_edge = True
        if add_edge:
            H.add_edge(edge_id, edge)

    return H


def check_beta_condition(beta, beta_minus, d):
    return (beta - beta_minus) >= (d - 1)

def build_HEDCS(hypergraph, beta, beta_minus):
    H = hnx.Hypergraph(hypergraph.incidence_dict)  # Initialize H to be equal to G
    degrees = {node: 0 for node in hypergraph.nodes}  # Initialize vertex degrees

    for edge in H.edges:
        for node in H.edges[edge]:
            degrees[node] += 1

    while True:
        violating_edge = None
        for edge in list(H.edges):
            edge_degree_sum = sum(degrees[node] for node in H.edges[edge])
            if edge_degree_sum > beta:
                violating_edge = edge
                H.remove_edge(violating_edge)
                for node in H.edges[violating_edge]:
                    degrees[node] -= 1
                break

        for edge in list(hypergraph.edges):
            if edge not in H.edges:
                edge_degree_sum = sum(degrees[node] for node in hypergraph.edges[edge])
                if edge_degree_sum < beta_minus:
                    violating_edge = edge
                    H.add_edge(violating_edge, hypergraph.edges[violating_edge])
                    for node in H.edges[violating_edge]:
                        degrees[node] += 1
                    break

        if violating_edge is None:
            break
    return H

def partition_hypergraph(hypergraph, k):
    edges = list(hypergraph.incidence_dict.items())
    random.shuffle(edges)
    partitions = [edges[i::k] for i in range(k)]
    return [hnx.Hypergraph(dict(part)) for part in partitions]
def HEDCS_matching(hypergraph: Hypergraph, s: int, debug=False) -> list:
    """
    Algorithm 3: HEDCS-Matching for Hypergraph Matching
    This algorithm constructs a Hyper-Edge Degree Constrained Subgraph (HEDCS) to find
    a maximal matching in a d-uniform hypergraph.

    Parameters:
    hypergraph (Hypergraph): A Hypergraph object.
    s (int): The amount of memory available per machine.

    Returns:
    list: The edges of the graph for the approximate matching.

    Raises:
    MemoryLimitExceededError: If the memory limit is exceeded during the matching process.

    Examples:
    >>> hypergraph = Hypergraph({0: (1, 2)})
    >>> result = HEDCS_matching(hypergraph, 10)
    >>> result is not None and all(len(edge) >= 2 for edge in result)
    True

    >>> hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    >>> result = HEDCS_matching(hypergraph, 10)
    >>> result is not None and all(len(edge) >= 2 for edge in result)
    True


    >>> edges = {'e1': [1, 2, 3], 'e2': [2, 3, 4], 'e3': [1, 4, 5]}
    >>> hypergraph = hnx.Hypergraph(edges)
    >>> s = 10
    >>> optimal_matching = ['e1']  # Assuming we know the optimal matching
    >>> approximate_matching = HEDCS_matching(hypergraph, s)
    >>> d = 3
    >>> len(approximate_matching) <= d**3 * len(optimal_matching)
    True


    >>> # Test with a larger hypergraph
    >>> edges_large = {f'e{i}': [i, i+1, i+2] for i in range(1, 101)}
    >>> hypergraph_large = Hypergraph(edges_large)
    >>> optimal_matching_large = [edges_large[f'e{i}'] for i in range(1, 101,3)]
    >>> approximate_matching_large = HEDCS_matching(hypergraph_large, s)
    >>> len(approximate_matching_large) <= d**3 * len(optimal_matching_large)
    True

    """
    edge_sizes = {len(edge) for edge in hypergraph.incidence_dict.values()}
    if len(edge_sizes) > 1:
        raise NonUniformHypergraphError("The hypergraph is not d-uniform.")

    d = next(iter(edge_sizes))
    n = len(hypergraph.nodes)
    m = len(hypergraph.edges)

    beta = 500 * d**3 * n**2 * (math.log(n)**3)
    gamma = 1 / (2 * n * math.log(n))
    k = int(m / (s * math.log(n)))
    beta_minus = (1 - gamma) * beta

    if not check_beta_condition(beta, beta_minus, d):
        raise ValueError(f"beta - beta_minus must be >= {d - 1}")

    # Partition the hypergraph
    partitions = partition_hypergraph(hypergraph, k)

    # Build HEDCS for each partition in parallel
    with ThreadPoolExecutor() as executor:
        HEDCS_list = list(executor.map(lambda part: build_HEDCS(part, beta, beta_minus), partitions))

    # Combine all the edges from the HEDCS subgraphs
    combined_edges = {}
    for H in HEDCS_list:
        combined_edges.update(H.incidence_dict)

    combined_hypergraph = hnx.Hypergraph(combined_edges)

    # Find the maximum matching in the combined hypergraph
    max_matching = maximal_matching(combined_hypergraph)

    return max_matching

if __name__ == '__main__':
    import doctest

    doctest.testmod()
