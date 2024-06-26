"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""
from datetime import time
from functools import lru_cache

import numpy as np
import hypernetx as hnx
from hypernetx.classes.hypergraph import Hypergraph
import math
import random
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def approximation_matching_checking(optimal: list, approx: list) -> bool:
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

def greedy_matching(hypergraph: Hypergraph, k: int) -> list:
    """
    Greedy algorithm for hypergraph matching
    This algorithm constructs a random k-partitioning of G and finds a maximal matching.

    Parameters:
    hypergraph (hnx.Hypergraph): A Hypergraph object.
    k (int): The number of partitions.

    Returns:
    list: The edges of the graph for the greedy matching.

    Example:
    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> edges = {'e1': [1, 2, 3], 'e2': [2, 3, 4], 'e3': [1, 4, 5]}
    >>> hypergraph = Hypergraph(edges)
    >>> k = 2
    >>> matching = greedy_matching(hypergraph, k)
    >>> matching
    [(2, 3, 4)]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> edges_large = {f'e{i}': list(range(i, i + 3)) for i in range(1, 50)}
    >>> hypergraph_large = Hypergraph(edges_large)
    >>> k = 5
    >>> matching_large = greedy_matching(hypergraph_large, k)
    >>> len(matching_large)
    12

    >>> edges_non_uniform = {'e1': [1, 2, 3], 'e2': [4, 5], 'e3': [6, 7, 8, 9]}
    >>> hypergraph_non_uniform = Hypergraph(edges_non_uniform)
    >>> try:
    ...     greedy_matching(hypergraph_non_uniform, k)
    ... except NonUniformHypergraphError:
    ...     print("NonUniformHypergraphError raised")
    NonUniformHypergraphError raised
    """
    logging.debug("Running Greedy Matching Algorithm")

    # Check if the hypergraph is empty
    if not hypergraph.incidence_dict:
        return []

    # Check if the hypergraph is d-uniform
    edge_sizes = {len(edge) for edge in hypergraph.incidence_dict.values()}
    if len(edge_sizes) > 1:
        raise NonUniformHypergraphError("The hypergraph is not d-uniform.")

    # Partition the hypergraph into k subgraphs
    partitions = partition_hypergraph(hypergraph, k)

    # Find maximum matching for each partition in parallel
    with ThreadPoolExecutor() as executor:
        MM_list = list(executor.map(maximal_matching, partitions))

    # Initialize the matching set
    M = set()

    # Process each partition's matching
    for MM_Gi in MM_list:
        # Add edges to M if they do not violate the matching property
        for edge in MM_Gi:
            if not any(set(edge) & set(matching_edge) for matching_edge in M):
                M.add(tuple(edge))

    return list(M)


class MemoryLimitExceededError(Exception):
    """Custom exception to indicate memory limit exceeded during hypergraph matching."""
    pass


class NonUniformHypergraphError(Exception):
    """Custom exception to indicate non d-uniform hypergraph during matching."""
    pass




# Helper functions
def edge_tuple(hypergraph):
    """Convert hypergraph edges to a hashable tuple."""
    return tuple((edge, tuple(sorted(hypergraph.edges[edge]))) for edge in sorted(hypergraph.edges))


@lru_cache(maxsize=None)
def cached_maximal_matching(edges):
    """Cached version of maximal matching calculation."""
    hypergraph = hnx.Hypergraph(dict(edges))
    matching = []
    matched_vertices = set()

    for edge in hypergraph.incidence_dict.values():
        if not any(vertex in matched_vertices for vertex in edge):
            matching.append(sorted(edge))
            matched_vertices.update(edge)
    return matching


def maximal_matching(hypergraph: Hypergraph) -> list:
    """Find a maximal matching in the hypergraph."""
    edges = edge_tuple(hypergraph)
    return cached_maximal_matching(edges)


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
    logging.debug(f"Sampled edges: {sampled_edges}")
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
    matching = maximal_matching(E_prime)
    logging.debug(f"Sampled hypergraph: {E_prime.incidence_dict}, Maximal matching: {matching}")
    return matching, E_prime


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
    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5)})
    >>> result = iterated_sampling(hypergraph, 1)
    >>> result
    [[2, 3, 4]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (2, 3, 4, 5), 2: (3, 4, 5, 6)})
    >>> result = iterated_sampling(hypergraph, 2)
    >>> result
    [[2, 3, 4, 5]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> result = None
    >>> try:
    ...     result = iterated_sampling(hypergraph, 0)  # Insufficient memory, expect failure
    ... except MemoryLimitExceededError:
    ...     pass
    >>> result is None
    True

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    >>> result = iterated_sampling(hypergraph, 10)  # Large enough memory, expect a result
    >>> result
    [[4, 5, 6], [1, 2, 3]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    >>> result = iterated_sampling(hypergraph, 3)
    >>> result
    [[2, 3, 4], [5, 6, 7]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 1), 4: (2, 6, 10, 14), 5: (3, 7, 11, 15), 6: (4, 8, 12, 1), 7: (5, 9, 13, 2), 8: (6, 10, 14, 3), 9: (7, 11, 15, 4)})
    >>> result = iterated_sampling(hypergraph, 4)
    >>> result
    [[4, 7, 11, 15], [2, 6, 10, 14]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> s = 10
    >>> edges_d4 = {'e1': [1, 2, 3, 4], 'e2': [2, 3, 4, 5], 'e3': [3, 4, 5, 6], 'e4': [4, 5, 6, 7]}
    >>> hypergraph_d4 = Hypergraph(edges_d4)
    >>> approximate_matching_d4 = iterated_sampling(hypergraph_d4, s)
    >>> approximate_matching_d4
    [[2, 3, 4, 5]]

    >>> edges_d5 = {'e1': [1, 2, 3, 4, 5], 'e2': [2, 3, 4, 5, 6], 'e3': [3, 4, 5, 6, 7]}
    >>> hypergraph_d5 = Hypergraph(edges_d5)
    >>> approximate_matching_d5 = iterated_sampling(hypergraph_d5, s)
    >>> approximate_matching_d5
    [[1, 2, 3, 4, 5]]

    >>> edges_d6 = {'e1': [1, 2, 3, 4, 5, 6], 'e2': [2, 3, 4, 5, 6, 7], 'e3': [3, 4, 5, 6, 7, 8]}
    >>> hypergraph_d6 = Hypergraph(edges_d6)
    >>> approximate_matching_d6 = iterated_sampling(hypergraph_d6, s)
    >>> approximate_matching_d6
    [[1, 2, 3, 4, 5, 6]]

    >>> edges_large = {f'e{i}': [i, i + 1, i + 2] for i in range(1, 101)}
    >>> hypergraph_large = Hypergraph(edges_large)
    >>> approximate_matching_large = iterated_sampling(hypergraph_large, s)
    >>> len(approximate_matching_large)
    26
    """
    logging.debug("Running Iterated Sampling Algorithm")

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
        logging.debug(f"After iteration {iterations}, matching: {M}")

        unmatched_vertices = set(S.nodes) - set(v for edge in M_prime for v in edge)
        induced_edges = [edge for edge in S.incidence_dict.values() if all(v in unmatched_vertices for v in edge)]
        if len(induced_edges) <= s:
            M.extend(maximal_matching(hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})))
            break
        S = hnx.Hypergraph({f'e{i}': tuple(edge) for i, edge in enumerate(induced_edges)})
        p = s / (5 * len(S.edges) * d) if len(S.edges) > 0 else 0

    if iterations >= max_iterations:
        raise MemoryLimitExceededError("Max iterations reached without finding a solution")

    logging.debug(f"Final matching result: {M}")
    return M


def check_beta_condition(beta, beta_minus, d):
    return (beta - beta_minus) >= (d - 1)


def build_HEDCS(hypergraph, beta, beta_minus):
    """
    Constructs a Hyper-Edge Degree Constrained Subgraph (HEDCS) from the given hypergraph G.

    Parameters:
    G (Hypergraph): The input hypergraph.
    beta (int): Degree threshold for adding edges.
    beta_minus (int): Complementary degree threshold for adding edges.

    Returns:
    Hypergraph: The constructed HEDCS.
    """
    H = hnx.Hypergraph(hypergraph.incidence_dict)  # Initialize H to be equal to G
    degrees = {node: 0 for node in hypergraph.nodes}  # Initialize vertex degrees

    for edge in H.edges:
        for node in H.edges[edge]:
            degrees[node] += 1

    logging.debug("Initial degrees: %s", degrees)

    while True:
        violating_edge = None
        for edge in list(H.edges):
            edge_degree_sum = sum(degrees[node] for node in H.edges[edge])
            if edge_degree_sum > beta:
                violating_edge = edge
                H.remove_edge(violating_edge)
                for node in H.edges[violating_edge]:
                    degrees[node] -= 1
                logging.debug(f"Removed edge {violating_edge} from HEDCS. Current degrees: {degrees}")
                break

        for edge in list(hypergraph.edges):
            if edge not in H.edges:
                edge_degree_sum = sum(degrees[node] for node in hypergraph.edges[edge])
                if edge_degree_sum < beta_minus:
                    violating_edge = edge
                    H.add_edge(violating_edge, hypergraph.edges[violating_edge])
                    for node in H.edges[violating_edge]:
                        degrees[node] += 1
                    logging.debug(f"Added edge {violating_edge} to HEDCS. Current degrees: {degrees}")
                    break

        if violating_edge is None:
            break
    logging.debug(f"Final HEDCS: {H.incidence_dict}")
    return H


def partition_hypergraph(hypergraph, k):
    edges = list(hypergraph.incidence_dict.items())
    random.shuffle(edges)
    partitions = [edges[i::k] for i in range(k)]
    logging.debug(f"Partitions: {partitions}")
    return [hnx.Hypergraph(dict(part)) for part in partitions]


def HEDCS_matching(hypergraph: Hypergraph, s: int) -> list:
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
    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2)})
    >>> result = HEDCS_matching(hypergraph, 10)
    >>> result
    [[1, 2]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    >>> result = HEDCS_matching(hypergraph, 10)
    >>> result
    [[1, 2], [3, 4]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> edges = {'e1': [1, 2, 3], 'e2': [2, 3, 4], 'e3': [1, 4, 5]}
    >>> hypergraph = Hypergraph(edges)
    >>> s = 10
    >>> approximate_matching = HEDCS_matching(hypergraph, s)
    >>> approximate_matching
    [[1, 2, 3]]

    >>> np.random.seed(42)
    >>> random.seed(42)
    >>> edges_large = {f'e{i}': [i, i + 1, i + 2] for i in range(1, 101)}
    >>> hypergraph_large = Hypergraph(edges_large)
    >>> approximate_matching_large = HEDCS_matching(hypergraph_large, s)
    >>> len(approximate_matching_large)
    34
    """
    logging.debug("Running HEDCS Matching Algorithm")

    edge_sizes = {len(edge) for edge in hypergraph.incidence_dict.values()}
    if len(edge_sizes) > 1:
        raise NonUniformHypergraphError("The hypergraph is not d-uniform.")

    d = next(iter(edge_sizes))
    n = len(hypergraph.nodes)
    m = len(hypergraph.edges)

    beta = 500 * d*3 * n*2 * (math.log(n)*3)
    gamma = 1 / (2 * n * math.log(n))
    k = math.ceil(m / (s * math.log(n)))
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

    logging.debug(f"Final HEDCS Matching result: {max_matching}")
    return max_matching


if __name__ == '__main__':
    import doctest

    doctest.testmod()