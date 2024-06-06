import random
import pytest
from hypernetx.classes.hypergraph import Hypergraph
from hypernetx.algorithms.matching_algorithms import greedy_d_approximation, maximal_matching
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



# def test_iterated_sampling():
#     random.seed(0)
#
#     test_hypergraphs = [
#         # Small hypergraph example 1
#         {
#             'name': 'Small Hypergraph 1',
#             'data': {
#                 'e0': (1, 2, 3),
#                 'e1': (4, 5, 6)
#             }
#         },
#         # Small hypergraph example 2
#         {
#             'name': 'Small Hypergraph 2',
#             'data': {
#                 'e0': (1, 2, 3),
#                 'e1': (2, 3, 4),
#                 'e2': (3, 4, 5),
#                 'e3': (5, 6, 7),
#                 'e4': (6, 7, 8),
#                 'e5': (7, 8, 9)
#             }
#         },
#         # Medium hypergraph example
#         {
#             'name': 'Medium Hypergraph',
#             'data': {
#                 'e0': (1, 2, 3, 4),
#                 'e1': (5, 6, 7, 8),
#                 'e2': (9, 10, 11, 12),
#                 'e3': (13, 14, 15, 1),
#                 'e4': (2, 6, 10, 14),
#                 'e5': (3, 7, 11, 15),
#                 'e6': (4, 8, 12, 1),
#                 'e7': (5, 9, 13, 2),
#                 'e8': (6, 10, 14, 3),
#                 'e9': (7, 11, 15, 4)
#             }
#         },
#         # Large hypergraph example
#         {
#             'name': 'Large Hypergraph',
#             'data': {
#                 f'e{i}': tuple(range(i * 4 + 1, i * 4 + 5)) for i in range(20)
#             }
#         }
#     ]
#
#     for hypergraph_info in test_hypergraphs:
#         name = hypergraph_info['name']
#         hypergraph_data = hypergraph_info['data']
#         print(f"\nTesting {name}...")
#         hypergraph = Hypergraph(hypergraph_data)
#         print(hypergraph_data)
#
#         for memory_limit in [5, 10, 15, 20, 25, 30]:
#             print(f"Memory limit: {memory_limit}")
#             approx_matching = iterated_sampling(hypergraph, 4, memory_limit)
#             baseline_matching = maximal_matching(hypergraph)
#
#             assert set(map(frozenset, approx_matching)) <= set(map(frozenset, baseline_matching)), \
#                 f"Failed for {name} with memory limit {memory_limit}"
#
#             print("Approximate Matching:", approx_matching)
#             print("Baseline Matching:", baseline_matching)
#             print("Is d-approximation:", set(map(frozenset, approx_matching)) <= set(map(frozenset, baseline_matching)))
#             print("-" * 50)



if __name__ == '__main__':
 pytest.main()