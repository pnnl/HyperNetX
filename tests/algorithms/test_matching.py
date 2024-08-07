"""
An implementation of the algorithms in:
"Distributed Algorithms for Matching in Hypergraphs", by Oussama Hanguir and Clifford Stein (2020), https://arxiv.org/abs/2009.09605v1
Programmer: Shira Rot, Niv
Date: 22.5.2024
"""


import pytest
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.algorithms.matching_algorithms import greedy_matching, HEDCS_matching, \
    MemoryLimitExceededError, approximation_matching_checking
from hypernetx.algorithms.matching_algorithms import iterated_sampling


def test_greedy_d_approximation_empty_input():
    """
    Test for an empty input hypergraph.
    """
    k = 2
    empty_hypergraph = Hypergraph({})
    assert greedy_matching(empty_hypergraph, k) == []


def test_greedy_d_approximation_small_inputs():
    """
    Test for small input hypergraphs.
    """
    k = 2
    hypergraph_1 = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}})
    assert greedy_matching(hypergraph_1, k) == [(1, 2, 3), (4, 5, 6)]

    hypergraph_2 = Hypergraph(
        {'e1': {1, 2, 3}, 'e2': {4, 5, 6}, 'e3': {7, 8, 9}, 'e4': {1, 4, 7}, 'e5': {2, 5, 8}, 'e6': {3, 6, 9}})
    result = greedy_matching(hypergraph_2, k)
    assert len(result) == 3
    assert all(edge in [(1, 2, 3), (4, 5, 6), (7, 8, 9)] for edge in result)


def test_greedy_d_approximation_large_input():
    """
    Test for a large input hypergraph.
    """
    k = 2
    large_hypergraph = Hypergraph({f'e{i}': {i, i + 1, i + 2} for i in range(1, 100, 3)})
    result = greedy_matching(large_hypergraph, k)
    assert len(result) == len(large_hypergraph.edges)
    assert all(edge in [(i, i + 1, i + 2) for i in range(1, 100, 3)] for edge in result)


def test_iterated_sampling_single_edge():
    """
    Test for a hypergraph with a single edge.
    It checks if the result is not None and if all edges in the result have at least 2 vertices.
    """
    hypergraph = Hypergraph({0: (1, 2, 3)})
    result = iterated_sampling(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)


def test_iterated_sampling_two_disjoint_edges():
    """
    Test for a hypergraph with two disjoint edges.
    It checks if the result is not None and if all edges in the result have at least 2 vertices.
    """
    hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    result = iterated_sampling(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)


def test_iterated_sampling_insufficient_memory():
    """
    Test for a hypergraph with insufficient memory.
    It checks if the function raises a MemoryLimitExceededError when memory is set to 0.
    """
    hypergraph = Hypergraph({0: (1, 2, 3)})
    with pytest.raises(MemoryLimitExceededError):
        iterated_sampling(hypergraph, 0)


def test_iterated_sampling_large_memory():
    """
    Test for a hypergraph with sufficient memory.
    It checks if the result is not None when memory is set to 10.
    """
    hypergraph = Hypergraph({0: (1, 2, 3), 1: (4, 5, 6)})
    result = iterated_sampling(hypergraph, 10)
    assert result is not None


def test_iterated_sampling_max_iterations():
    """
    Test for a hypergraph reaching maximum iterations.
    """
    hypergraph = Hypergraph({0: (1, 2, 3), 1: (2, 3, 4), 2: (3, 4, 5), 3: (5, 6, 7), 4: (6, 7, 8), 5: (7, 8, 9)})
    result = iterated_sampling(hypergraph, 3)
    assert result is None or all(len(edge) >= 2 for edge in result)


def test_iterated_sampling_large_hypergraph():
    """
    Test for a large hypergraph.
    """
    edges_large = {f'e{i}': [i, i + 1, i + 2] for i in range(1, 101)}
    hypergraph_large = Hypergraph(edges_large)
    optimal_matching_large = [edges_large[f'e{i}'] for i in range(1, 101, 3)]
    result = iterated_sampling(hypergraph_large, 10)
    assert result is not None and approximation_matching_checking(optimal_matching_large, result)


def test_HEDCS_matching_single_edge():
    """
    Test for a hypergraph with a single edge.
    """
    hypergraph = Hypergraph({0: (1, 2)})
    result = HEDCS_matching(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)


def test_HEDCS_matching_two_edges():
    """
    Test for a hypergraph with two disjoint edges.
    """
    hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    result = HEDCS_matching(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)


def test_HEDCS_matching_with_optimal_matching():
    """
    Test with a hypergraph where the optimal matching is known.
    """
    edges = {'e1': [1, 2, 3], 'e2': [2, 3, 4], 'e3': [1, 4, 5]}
    hypergraph = Hypergraph(edges)
    s = 10
    optimal_matching = [[1, 2, 3]]  # Assuming we know the optimal matching
    approximate_matching = HEDCS_matching(hypergraph, s)
    assert approximation_matching_checking(optimal_matching, approximate_matching)


def test_HEDCS_matching_large_hypergraph():
    """
    Test with a larger hypergraph.
    """
    edges_large = {f'e{i}': [i, i + 1, i + 2] for i in range(1, 101)}
    hypergraph_large = Hypergraph(edges_large)
    s = 10
    optimal_matching_large = [edges_large[f'e{i}'] for i in range(1, 101, 3)]
    approximate_matching_large = HEDCS_matching(hypergraph_large, s)
    assert approximation_matching_checking(optimal_matching_large, approximate_matching_large)


if __name__ == '__main__':
    pytest.main()