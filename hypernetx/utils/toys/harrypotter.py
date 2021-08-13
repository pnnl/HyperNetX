import os
import hypernetx as hnx
from collections import OrderedDict, defaultdict
import scipy
from scipy.sparse import coo_matrix, issparse
import pandas as pd
import numpy as np
import itertools as it

__all__ = ["HarryPotter"]

current_dir = os.path.dirname(os.path.abspath(__file__))


class HarryPotter(object):
    def __init__(self, cols=None):

        # Read dataset in using pandas. Fix index column or use default pandas index.

        try:
            fname = "https://raw.githubusercontent.com/pnnl/HyperNetX/master/hypernetx/utils/toys/HarryPotter_Characters.csv"
            harrydata = pd.read_csv(fname, encoding="unicode_escape")
        except:
            fname = f"{current_dir}/HarryPotter_Characters.csv"
            harrydata = pd.read_csv(fname, encoding="unicode_escape")

        self.harrydata = pd.DataFrame(harrydata)

        # Choose string to fill NaN. These will be set to 0 in system id = sid
        columns = cols or [
            "House",
            "Blood status",
            "Species",
            "Hair colour",
            "Eye colour",
        ]
        harry = harrydata[columns].fillna("Unknown")
        for c in harry.columns:
            harry[c] = (
                harry[c]
                .apply(lambda x: x.replace("\xa0", " "))
                .apply(lambda x: x.replace("Unknown", f"Unknown {c}"))
            )
        self.dataframe = harry

        ctr = [hnx.HNXCount() for c in range(len(columns))]
        ldict = OrderedDict()
        rdict = OrderedDict()
        for idx, c in enumerate(columns):
            ldict[c] = defaultdict(ctr[idx])
            rdict[c] = OrderedDict()
            ldict[c][f"Unknown {c}"]
            rdict[c][0] = f"Unknown {c}"
            for k in harry[c]:
                ldict[c][k]
                rdict[c][ldict[c][k]] = k
            ldict[c] = dict(ldict[c])
        self.dims = dims = tuple([len(ldict[c]) for c in harry.columns])

        # ### Create an array of tuples giving positions of 1's in incidence Tensor
        # - The tuples indicate one point across the possible node/edge assignments
        # - The dimensions of the tuple give the number of unique labels in potential nodes/columns

        m = len(harry)
        n = len(harry.columns)
        data = np.zeros((m, n), dtype=int)
        for rid in range(m):
            for cid in range(n):
                c = harry.columns[cid]
                data[rid, cid] = ldict[c][harry.iloc[rid][c]]

        self.data, self.counts = hnx.remove_row_duplicates(data, aggregateby="count")
        # Create incidence Tensor and labels
        imat = np.zeros(dims, dtype=int)
        for d in self.data:
            imat[tuple(d)] = self.counts[tuple(d)]
        self.arr = imat

        slabels = OrderedDict()
        for cdx, c in enumerate(list(ldict.keys())):
            slabels.update({c: np.array(list(ldict[c].keys()))})
        self.labels = slabels
