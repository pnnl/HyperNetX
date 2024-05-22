import pytest
from hypernetx.algorithms.matching_algorithms import greedy_d_approximation, hedcs_based_approximation, iterated_sampling

def test_greedy_d_approximation():
    # Test for empty input
    assert greedy_d_approximation([], 3) == 0

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        greedy_d_approximation([(1, 2, 3)], 1)

    # Test small inputs
    hypergraph = [(1, 2, 3), (4, 5, 6)]
    assert greedy_d_approximation(hypergraph, 3) == 2

    hypergraph = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 4, 7), (2, 5, 8), (3, 6, 9)]
    assert greedy_d_approximation(hypergraph, 3) == 3

    hypergraph = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (5, 6, 7), (6, 7, 8), (7, 8, 9)]
    assert greedy_d_approximation(hypergraph, 3) == 2

    # Test large input
    large_hypergraph = [(i, i+1, i+2) for i in range(1, 1000, 3)]
    assert greedy_d_approximation(large_hypergraph, 3) == 333

def test_hedcs_based_approximation():
    # Test for empty input
    assert hedcs_based_approximation([], 3, 0.1) == 0

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        hedcs_based_approximation([(1, 2, 3)], 1, 0.1)

    # Test small inputs
    hypergraph = [(1, 2, 3), (4, 5, 6)]
    assert hedcs_based_approximation(hypergraph, 3, 0.1) == 2

    hypergraph = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 4, 7), (2, 5, 8), (3, 6, 9)]
    assert hedcs_based_approximation(hypergraph, 3, 0.1) == 3

    hypergraph = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (5, 6, 7), (6, 7, 8), (7, 8, 9)]
    assert hedcs_based_approximation(hypergraph, 3, 0.1) == 2

    # Test large input
    large_hypergraph = [(i, i+1, i+2) for i in range(1, 1000, 3)]
    assert hedcs_based_approximation(large_hypergraph, 3, 0.1) == 333

def test_iterated_sampling():
    # Test for empty input
    assert iterated_sampling([], 3) == 0

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        iterated_sampling([(1, 2, 3)], 1)

    # Test small inputs
    hypergraph = [(1, 2, 3), (4, 5, 6)]
    assert iterated_sampling(hypergraph, 3) == 2

    hypergraph = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 4, 7), (2, 5, 8), (3, 6, 9)]
    assert iterated_sampling(hypergraph, 3) == 3

    hypergraph = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (5, 6, 7), (6, 7, 8), (7, 8, 9)]
    assert iterated_sampling(hypergraph, 3) == 2

    # Test large input
    large_hypergraph = [(i, i+1, i+2) for i in range(1, 1000, 3)]
    assert iterated_sampling(large_hypergraph, 3) == 333

if __name__ == '__main__':
    pytest.main()
