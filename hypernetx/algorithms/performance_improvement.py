import time
import hypernetx as hnx
from hypernetx.classes.hypergraph import Hypergraph
from functools import lru_cache
import random


# Definitions of exceptions
class MemoryLimitExceededError(Exception):
    """Exception to indicate memory limit exceeded during hypergraph matching."""
    pass


class NonUniformHypergraphError(Exception):
    """Exception to indicate non d-uniform hypergraph during matching."""
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


def create_random_hypergraph(num_edges, num_vertices, edge_size):
    """Generate a random hypergraph for testing."""
    edges = {}
    for i in range(num_edges):
        vertices = random.sample(range(num_vertices), edge_size)
        edges[f'e{i}'] = vertices
    return Hypergraph(edges)


# Timing and running the algorithms
def run_algorithm(algorithm, hypergraph):
    start = time.time()
    result = algorithm(hypergraph)
    end = time.time()
    return result, end - start


def main():
    num_edges = 100
    num_vertices = 50
    edge_size = 5

    # Create a random hypergraph
    hypergraph = create_random_hypergraph(num_edges, num_vertices, edge_size)

    # Run both algorithms
    _, time_original = run_algorithm(maximal_matching, hypergraph)
    _, time_cached = run_algorithm(maximal_matching, hypergraph)  # Run twice to show cached advantage

    print(f"Original Maximal Matching Time: {time_original} seconds")
    print(f"Cached Maximal Matching Time: {time_cached} seconds")


if __name__ == '__main__':
    main()