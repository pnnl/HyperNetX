import numpy as np
import hypernetx as hnx
from hypernetx.classes.hypergraph import Hypergraph
import math
import random
import time
import experiments_csv
import pandas as pd
import logging
from matplotlib import pyplot as plt
import seaborn as sns

from hypernetx.algorithms.matching_algorithms import (
    maximal_matching,
    sample_edges,
    iterated_sampling,
    HEDCS_matching,
    MemoryLimitExceededError,
    NonUniformHypergraphError,
    partition_hypergraph,
    greedy_matching,
    logger as matching_logger
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# matching_logger.setLevel(logging.DEBUG)

# Function to generate random d-uniform hypergraphs
def generate_random_hypergraph(n, d, m):
    edges = {f'e{i}': random.sample(range(1, n+1), d) for i in range(m)}
    return Hypergraph(edges)

def run_experiment(algorithm, n, d, m, s):
    hypergraph = generate_random_hypergraph(n, d, m)

    logger.info(f"Running {algorithm.__name__} with n={n}, d={d}, m={m}, s={s}")
    start_time = time.time()
    matching = algorithm(hypergraph, s)
    end_time = time.time()

    match_size = len(matching)
    run_time = end_time - start_time

    logger.info(f"Finished {algorithm.__name__} with match_size={match_size}, run_time={run_time:.4f} seconds")
    return {
        "algorithm": algorithm.__name__,
        "n": n,
        "d": d,
        "m": m,
        "s": s,
        "match_size": match_size,
        "run_time": run_time
    }

def define_experiment():
    experiment = experiments_csv.Experiment("results/", "hypergraph_matching.csv")
    experiment.logger.setLevel(logging.INFO)

    sizes = [100, 200, 400, 800, 1600]
    d = 3
    m = 100
    s = 10

    input_ranges = {
        "algorithm": [iterated_sampling, HEDCS_matching, greedy_matching],
        "n": sizes,
        "d": [d],
        "m": [m],
        "s": [15, 20]
    }
    experiment.run(run_experiment, input_ranges)

    return experiment

if __name__ == "__main__":
    experiment = define_experiment()

    # Draw results
    df = experiment.dataFrame

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))

    sns.lineplot(data=df, x="s", y="run_time", hue="algorithm", marker="o")
    plt.title("Running Time of Hypergraph Matching Algorithms")
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Running Time (seconds)")
    plt.savefig("running_time_comparison.png")
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="s", y="match_size", hue="algorithm", marker="o")
    plt.title("Matching Size of Hypergraph Matching Algorithms")
    plt.xlabel("Number of Vertices (n)")
    plt.ylabel("Matching Size")
    plt.savefig("matching_size_comparison.png")
    plt.show()
