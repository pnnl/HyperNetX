from collections import defaultdict
from hypernetx import Hypergraph
import pandas as pd
import pickle

def read_csv(filename, index1, index2, name="_"):
    df = pd.read_csv(filename)
    return read_df(df, index1, index2, name)

def read_weighted_csv(filename, index1, index2, weightindex, name="_"):
    df = pd.read_csv(filename)
    return read_weighted_df(df, index1, index2, weightindex, name)

def read_df(df, index1, index2, name="_"):
    edges = defaultdict(list)
    # add error handling
    for row in df.itertuples():
        edges[row[index1]].append(row[index2])
    return Hypergraph(edges, name)

# currently not finished.
def read_weighted_df(df, index1, index2, weightindex, name="_"):
    edges = defaultdict(list)
    for row in df.itertuples():
        edges[row[index1]].append(row[index2])
    return Hypergraph(edges, name)

def read_numpy_array():
    return 0

def to_pickle(obj, filename):
    """Writes object to a pickle file"""
    with open(f"{filename}", "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filepath):
    """Returns object from file"""
    with open(filepath, "rb") as f:
        temp = pickle.load(f)
    return temp
