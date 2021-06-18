from collections import defaultdict
from hypernetx import Hypergraph, Entity
import pandas as pd
import csv

# read ScHOLP data?

def read_csv(filename, index1=0, index2=1, name="_", header=None):
    df = pd.read_csv(filename, header=header)
    return read_df(df, index1+1, index2+1, name, header=header)

def read_df(df, index1=1, index2=2, name="_"):
    edges = defaultdict(list)
    # add error handling
    for row in df.itertuples():
        edges[row[index1]].append(row[index2])
    return Hypergraph(edges, name)

# can accept a numpy array or a dataframe
def read_incidence_matrix(I):
    return 0

# accepts a list of iterables
def read_edgelist(filename, name="_", delimiter = ",", isweighted=False):
    with open(filename) as file:
        reader = csv.reader(file)

    edgelist = list()
    uid = 0
    for line in reader:
        # convert number string to int unless it's not a number
        data = [int(x) for x in line.split(delimiter) if x.strip().isdigit()]
        if isweighted:
            edgelist.append(Entity(uid, data[:-1], weight=float(data[-1])))
        else:
            edgelist.append(data)
    return Hypergraph(edgelist, name=name)

# def from_numpy_array(
#         cls,
#         M,
#         node_names=None,
#         edge_names=None,
#         node_label="nodes",
#         edge_label="edges",
#         name=None,
#         key=None,
#         static=False,
#         use_nwhy=False,
#     ):
#         """
#         Create a hypergraph from a real valued matrix represented as a 2 dimensionsl numpy array.
#         The matrix is converted to a matrix of 0's and 1's so that any truthy cells are converted to 1's and
#         all others to 0's.

#         Parameters
#         ----------
#         M : real valued array-like object, 2 dimensions
#             representing a real valued matrix with rows corresponding to nodes and columns to edges

#         node_names : object, array-like, default=None
#             List of node names must be the same length as M.shape[0].
#             If None then the node names correspond to row indices with 'v' prepended.

#         edge_names : object, array-like, default=None
#             List of edge names must have the same length as M.shape[1].
#             If None then the edge names correspond to column indices with 'e' prepended.

#         name : hashable

#         key : (optional) function
#             boolean function to be evaluated on each cell of the array,
#             must be applicable to numpy.array

#         Returns
#         -------
#          : Hypergraph

#         Note
#         ----
#         The constructor does not generate empty edges.
#         All zero columns in M are removed and the names corresponding to these
#         edges are discarded.


#         """
#         # Create names for nodes and edges
#         # Validate the size of the node and edge arrays

#         M = np.array(M)
#         if len(M.shape) != (2):
#             raise HyperNetXError("Input requires a 2 dimensional numpy array")
#         # apply boolean key if available
#         if key:
#             M = key(M)

#         if node_names is not None:
#             nodenames = np.array(node_names)
#             if len(nodenames) != M.shape[0]:
#                 raise HyperNetXError(
#                     "Number of node names does not match number of rows."
#                 )
#         else:
#             nodenames = np.array([f"v{idx}" for idx in range(M.shape[0])])

#         if edge_names is not None:
#             edgenames = np.array(edge_names)
#             if len(edgenames) != M.shape[1]:
#                 raise HyperNetXError(
#                     "Number of edge_names does not match number of columns."
#                 )
#         else:
#             edgenames = np.array([f"e{jdx}" for jdx in range(M.shape[1])])

#         if static or use_nwhy:
#             arr = np.array(M)
#             if key:
#                 arr = key(arr) * 1
#             arr = arr.transpose()
#             labels = OrderedDict([(edge_label, edgenames), (node_label, nodenames)])
#             E = StaticEntitySet(arr=arr, labels=labels)
#             return Hypergraph(E, name=name, use_nwhy=use_nwhy)

#         else:
#             # Remove empty column indices from M columns and edgenames
#             colidx = np.array([jdx for jdx in range(M.shape[1]) if any(M[:, jdx])])
#             colidxsum = np.sum(colidx)
#             if not colidxsum:
#                 return Hypergraph()
#             else:
#                 M = M[:, colidx]
#                 edgenames = edgenames[colidx]
#                 edict = dict()
#                 # Create an EntitySet of edges from M
#                 for jdx, e in enumerate(edgenames):
#                     edict[e] = nodenames[
#                         [idx for idx in range(M.shape[0]) if M[idx, jdx]]
#                     ]
#                 return Hypergraph(edict, name=name)

#     @classmethod
#     def from_dataframe(
#         cls,
#         df,
#         columns=None,
#         rows=None,
#         name=None,
#         fillna=0,
#         transpose=False,
#         transforms=[],
#         key=None,
#         node_label="nodes",
#         edge_label="edges",
#         static=False,
#         use_nwhy=False,
#     ):
#         """
#         Create a hypergraph from a Pandas Dataframe object using index to label vertices
#         and Columns to label edges. The values of the dataframe are transformed into an 
#         incidence matrix.  
#         Note this is different than passing a dataframe directly
#         into the Hypergraph constructor. The latter automatically generates a static hypergraph
#         with edge and node labels given by the cell values.

#         Parameters
#         ----------
#         df : Pandas.Dataframe
#             a real valued dataframe with a single index

#         columns : (optional) list, default = None
#             restricts df to the columns with headers in this list.

#         rows : (optional) list, default = None
#             restricts df to the rows indexed by the elements in this list.

#         name : (optional) string, default = None

#         fillna : float, default = 0
#             a real value to place in empty cell, all-zero columns will not generate
#             an edge.

#         transpose : (optional) bool, default = False
#             option to transpose the dataframe, in this case df.Index will label the edges
#             and df.columns will label the nodes, transpose is applied before transforms and
#             key

#         transforms : (optional) list, default = []
#             optional list of transformations to apply to each column,
#             of the dataframe using pd.DataFrame.apply().
#             Transformations are applied in the order they are
#             given (ex. abs). To apply transforms to rows or for additional
#             functionality, consider transforming df using pandas.DataFrame methods
#             prior to generating the hypergraph.

#         key : (optional) function, default = None
#             boolean function to be applied to dataframe. Must be defined on numpy
#             arrays.

#         See also
#         --------
#         from_numpy_array())


#         Returns
#         -------
#         : Hypergraph

#         Notes
#         -----
#         The `from_dataframe` constructor does not generate empty edges.
#         All-zero columns in df are removed and the names corresponding to these
#         edges are discarded.
#         Restrictions and data processing will occur in this order:

#             1. column and row restrictions
#             2. fillna replace NaNs in dataframe
#             3. transpose the dataframe
#             4. transforms in the order listed
#             5. boolean key

#         This method offers the above options for wrangling a dataframe into an incidence
#         matrix for a hypergraph. For more flexibility we recommend you use the Pandas
#         library to format the values of your dataframe before submitting it to this
#         constructor.

#         """

#         if type(df) != pd.core.frame.DataFrame:
#             raise HyperNetXError("Error: Input object must be a pandas dataframe.")

#         if columns:
#             df = df[columns]
#         if rows:
#             df = df.loc[rows]

#         df = df.fillna(fillna)
#         if transpose:
#             df = df.transpose()

#         # node_names = np.array(df.index)
#         # edge_names = np.array(df.columns)

#         for t in transforms:
#             df = df.apply(t)
#         if key:
#             mat = key(df.values) * 1
#         else:
#             mat = df.values * 1

#         params = {
#             "node_names": np.array(df.index),
#             "edge_names": np.array(df.columns),
#             "name": name,
#             "node_label": node_label,
#             "edge_label": edge_label,
#             "static": static,
#             "use_nwhy": use_nwhy,
#         }
#         return cls.from_numpy_array(mat, **params)