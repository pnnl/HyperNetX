import random
import pytest
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.algorithms.matching_algorithms import greedy_d_approximation, maximal_matching, HEDCS_matching
from  hypernetx.algorithms.matching_algorithms import iterated_sampling


def test_greedy_d_approximation():
    # Test for empty input
    empty_hypergraph = Hypergraph({})
    assert greedy_d_approximation(empty_hypergraph) == []

    # Test for wrong input (d is less than 2)
    hypergraph_with_small_edge = Hypergraph({'e1': {1}})
    with pytest.raises(ValueError):
        greedy_d_approximation(hypergraph_with_small_edge)

    # Test small inputs
    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}})
    assert greedy_d_approximation(hypergraph) == [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}]

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}, 'e3': {7, 8, 9}, 'e4': {1, 4, 7}, 'e5': {2, 5, 8}, 'e6': {3, 6, 9}})
    result = greedy_d_approximation(hypergraph)
    assert len(result) == 3
    assert all(edge in [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}, {'e3': {7, 8, 9}}] for edge in result)

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {2, 3, 4}, 'e3': {3, 4, 5}, 'e4': {5, 6, 7}, 'e5': {6, 7, 8}, 'e6': {7, 8, 9}})
    result = greedy_d_approximation(hypergraph)
    assert len(result) == 2
    assert all(edge in [{'e1': {1, 2, 3}}, {'e4': {5, 6, 7}}, {'e2': {2, 3, 4}}, {'e5': {6, 7, 8}}, {'e3': {3, 4, 5}}, {'e6': {7, 8, 9}}] for edge in result)

    # Test large input
    large_hypergraph = Hypergraph({f'e{i}': {i, i+1, i+2} for i in range(1, 100, 3)})
    result = greedy_d_approximation(large_hypergraph)
    assert len(result) == len(large_hypergraph.edges)
    assert all(edge in [{f'e{i}': {i, i+1, i+2}} for i in range(1, 100, 3)] for edge in result)



def test_iterated_sampling():
    # Test for a hypergraph with a single edge
    hypergraph = Hypergraph({0: (1, 2, 3)})
    result = iterated_sampling(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)

    # Test for a hypergraph with two disjoint edges
    hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    result = iterated_sampling(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)



def test_HEDCS_matching():
    # Test for a hypergraph with a single edge
    hypergraph = Hypergraph({0: (1, 2)})
    result = HEDCS_matching(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)

    # Test for a hypergraph with two disjoint edges
    hypergraph = Hypergraph({0: (1, 2), 1: (3, 4)})
    result = HEDCS_matching(hypergraph, 10)
    assert result is not None and all(len(edge) >= 2 for edge in result)



if __name__ == '__main__':
 pytest.main()