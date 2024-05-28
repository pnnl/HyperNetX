import pytest
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.algorithms.matching_algorithms import greedy_d_approximation, hedcs_based_approximation, iterated_sampling

def test_greedy_d_approximation():
    # Test for empty input
    empty_hypergraph = Hypergraph({})
    assert greedy_d_approximation(empty_hypergraph, 3) == []

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        hypergraph = Hypergraph({'e1': {1, 2, 3}})
        greedy_d_approximation(hypergraph, 1)

    # Test small inputs
    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}})
    assert greedy_d_approximation(hypergraph, 3) == [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}]

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}, 'e3': {7, 8, 9}, 'e4': {1, 4, 7}, 'e5': {2, 5, 8}, 'e6': {3, 6, 9}})
    result = greedy_d_approximation(hypergraph, 3)
    assert len(result) == 3
    assert all(edge in [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}, {'e3': {7, 8, 9}}] for edge in result)

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {2, 3, 4}, 'e3': {3, 4, 5}, 'e4': {5, 6, 7}, 'e5': {6, 7, 8}, 'e6': {7, 8, 9}})
    result = greedy_d_approximation(hypergraph, 3)
    assert len(result) == 2
    assert all(edge in [{'e1': {1, 2, 3}}, {'e4': {5, 6, 7}}, {'e2': {2, 3, 4}}, {'e5': {6, 7, 8}}, {'e3': {3, 4, 5}}, {'e6': {7, 8, 9}}] for edge in result)

    # Test large input
    large_hypergraph = Hypergraph({f'e{i}': {i, i+1, i+2} for i in range(1, 100, 3)})
    result = greedy_d_approximation(large_hypergraph, 3)
    assert len(result) == len(large_hypergraph.edges)
    assert all(edge in [{f'e{i}': {i, i+1, i+2}} for i in range(1, 100, 3)] for edge in result)



def test_hedcs_based_approximation():
    # Test for empty input
    empty_hypergraph = Hypergraph({})
    assert hedcs_based_approximation(empty_hypergraph, 3, 1) == []

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        hypergraph = Hypergraph({'e1': {1, 2, 3}})
        hedcs_based_approximation(hypergraph, 1, 1)

    # Test small inputs
    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}})
    result = hedcs_based_approximation(hypergraph, 3, 2)
    assert len(result) == 2
    assert all(edge in [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}] for edge in result)

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}, 'e3': {7, 8, 9}, 'e4': {1, 4, 7}, 'e5': {2, 5, 8}, 'e6': {3, 6, 9}})
    result = hedcs_based_approximation(hypergraph, 3, 3)
    assert len(result) == 3
    assert all(edge in [{'e1': {1, 2, 3}}, {'e2': {4, 5, 6}}, {'e3': {7, 8, 9}}] for edge in result)

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {2, 3, 4}, 'e3': {3, 4, 5}, 'e4': {5, 6, 7}, 'e5': {6, 7, 8}, 'e6': {7, 8, 9}})
    result = hedcs_based_approximation(hypergraph, 3, 2)
    assert len(result) == 2
    assert all(edge in [{'e1': {1, 2, 3}}, {'e4': {5, 6, 7}}, {'e2': {2, 3, 4}}, {'e5': {6, 7, 8}}, {'e3': {3, 4, 5}}, {'e6': {7, 8, 9}}] for edge in result)

    # Test large input
    large_hypergraph = Hypergraph({f'e{i}': {i, i+1, i+2} for i in range(1, 100, 3)})
    result = hedcs_based_approximation(large_hypergraph, 3, 10)
    assert len(result) == 33
    assert all(edge in [{f'e{i}': {i, i+1, i+2}} for i in range(1, 100, 3)] for edge in result)


def test_iterated_sampling():
    # Test for empty input
    empty_hypergraph = Hypergraph({})
    assert iterated_sampling(empty_hypergraph, 3) == 0

    # Test for wrong input (d is less than 2)
    with pytest.raises(ValueError):
        hypergraph = Hypergraph({'e1': {1, 2, 3}})
        iterated_sampling(hypergraph, 1)

    # Test small inputs
    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}})
    assert iterated_sampling(hypergraph, 3) == 2

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {4, 5, 6}, 'e3': {7, 8, 9}, 'e4': {1, 4, 7}, 'e5': {2, 5, 8}, 'e6': {3, 6, 9}})
    assert iterated_sampling(hypergraph, 3) == 3

    hypergraph = Hypergraph({'e1': {1, 2, 3}, 'e2': {2, 3, 4}, 'e3': {3, 4, 5}, 'e4': {5, 6, 7}, 'e5': {6, 7, 8}, 'e6': {7, 8, 9}})
    assert iterated_sampling(hypergraph, 3) == 2

    # Test large input
    large_hypergraph = Hypergraph({f'e{i}': {i, i+1, i+2} for i in range(1, 1000, 3)})
    assert iterated_sampling(large_hypergraph, 3) == 333

if __name__ == '__main__':
    pytest.main()
