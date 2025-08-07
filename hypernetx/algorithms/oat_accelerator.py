"""
Homology accelerated with Open Applied Topology (OAT)
=====================================================
This module uses `Open Applied Topology (OAT) <https://openappliedtopology.github.io>`_
as a backend to accelerate homology computations.
OAT is a pip-installable package with no special hardware or software requirements.

This module computes the homology of a hypergraph with coefficients in
the field of rational numbers. It provides access to betti numbers and cycle
representatives, in addition to several plotting functions.
"""

import hypernetx as hnx
import numpy as np
import pandas as pd
import copy
import plotly.graph_objects as go
import oat_python as oat


class AssociatedSimplicialComplexHomology:
    def __init__(
        self,
        max_homology_dimension,
        cycle_representatives_with_numeric_node_labels,
        node_label_dict,
    ):
        self.cycle_representatives_with_numeric_node_labels = (
            cycle_representatives_with_numeric_node_labels
        )
        self.node_label_dict = node_label_dict
        self.max_homology_dimension = max_homology_dimension

    def cycle_representatives(
        self, numeric_node_labels=False, count_simplices_per_cycle=False
    ):
        """
        Returns a basis of cycle representatives for the associated simplicial complex of a hypergraph.

        Parameters
        ----------
        numeric_node_labels : bool, optional
            If `True`, then the label used for each node will be an integer 
            corresponding to the position of that node in `list(H.nodes())`. 
            Otherwise nodes retain the same labels they hold in the hypergraph. 
            Defaults to `False`.
        count_simplices_per_cycle : bool, optional
            If `True`, then the data frame returned will have a column labeled 
            `nnz`, which stores the number of simplices in each cycle representative.
            Defaults to `False`.

        Returns
        -------
        pandas.DataFrame
            A dataframe where each row corresponds to a cycle representative. The columns are:
            
            - `dimension`: the homological dimension of the cycle representative
            - `cycle representative`: the cycle representative, formatted as a pandas.DataFrame
            - (optionally) `nnz`: the number of simplices in the cycle representative

        Examples
        --------
        Here we define a hypergraph representing a three-edge cycle graph and an 
        isolated vertex `d`, then compute its homology and cycle representatives.

        >>> import hypernetx
        >>> from hypernetx.algorithms import oat_accelerator
        >>> H                           =   hypernetx.Hypergraph([['a','b'], ['b','c'], ['c','a'], ['d']])
        >>> homology                    =   oat_accelerator.get_homology(H, max_homology_dimension=2)
        >>> cycle_representatives       =   homology.cycle_representatives()
        >>> print(cycle_representatives)
        
        Expected Output::
        
            dimension    cycle representative
            0           simplex coefficient 0 [d] 1
            0           simplex coefficient 0 [a] 1
            1           simplex coefficient 0 [b, c] 1 1 ...

        >>> print(cycle_representatives["cycle representative"][0])
        
        Expected Output::
        
            simplex    coefficient
            [d]        1

        >>> print(cycle_representatives["cycle representative"][1])
        
        Expected Output::
        
            simplex    coefficient
            [a]        1

        >>> print(cycle_representatives["cycle representative"][2])
        
        Expected Output::
        
            simplex    coefficient
            [b, c]     1
            [a, c]    -1
            [a, b]     1
        """

        if not hasattr(self, "cycle_representatives_with_numeric_node_labels"):
            print(
                "Cycle representatives have not been computed. Try running `get_homology`."
            )
            return

        cycles = copy.deepcopy(self.cycle_representatives_with_numeric_node_labels)
        if not count_simplices_per_cycle:
            del cycles["nnz"]

        if numeric_node_labels:
            return cycles
        else:
            relabeled_cycles = []
            for cycle in cycles["cycle representative"]:
                cycle = copy.deepcopy(
                    cycle
                )  # we determined experimentally that this deepcopy is necessary
                relabeled_simplices = [
                    [self.node_label_dict[i] for i in simplex]
                    for simplex in cycle["simplex"]
                ]
                cycle["simplex"] = relabeled_simplices
                relabeled_cycles.append(cycle)
            cycles["cycle representative"] = relabeled_cycles
            return cycles

    def betti_numbers(self):
        histo = [0 for _ in range(1 + self.max_homology_dimension)]
        for p in self.cycle_representatives_with_numeric_node_labels["dimension"]:
            histo[p] += 1
        df = pd.DataFrame({"Betti number": histo})
        df.index.name = "homology dimension"
        return df


def get_homology(h, max_homology_dimension=0):
    """
    Computes the homology of the associated simplicial complex, with rational coefficients.

    Concretely, returns the homology of the abstract simplicial complex `S` whose 
    simplices are the subsets of the hyperedges of the hypergraph.

    Parameters
    ----------
    h : hypernetx.Hypergraph
        A hypergraph.
    max_homology_dimension : int, optional
        Homology will be computed up to this dimension. Defaults to 0.

    Returns
    -------
    hypernetx.AssociatedSimplicialComplexHomology
        An object that can be queried to obtain betti numbers and cycle representatives.

    Examples
    --------
    Here we define a hypergraph representing a three-edge cycle graph and an 
    isolated vertex `d`, then compute its homology and cycle representatives.

    >>> import hypernetx
    >>> from hypernetx.algorithms import oat_accelerator
    >>> H                           =   hypernetx.Hypergraph([['a','b'], ['b','c'], ['c','a'], ['d']])
    >>> homology                    =   oat_accelerator.get_homology(H, max_homology_dimension=2)
    >>> cycle_representatives       =   homology.cycle_representatives()
    >>> print(cycle_representatives)

    Expected Output::

        dimension    cycle representative
        0           simplex coefficient 0 [d] 1
        0           simplex coefficient 0 [a] 1
        1           simplex coefficient 0 [b, c] 1 1 ...

    >>> print(cycle_representatives["cycle representative"][0])
    
    Expected Output::

        simplex    coefficient
        [d]        1

    >>> print(cycle_representatives["cycle representative"][1])
    
    Expected Output::

        simplex    coefficient
        [a]        1

    >>> print(cycle_representatives["cycle representative"][2])
    
    Expected Output::

        simplex    coefficient
        [b, c]     1
        [a, c]    -1
        [a, b]     1
    """
    # format hypergraph as list of lists
    (csr, node_label_sequence, edge_labels) = h.incidence_matrix(index=True)
    csr = (
        csr.transpose().tocsr()
    )  # note that we have to transpose !!! probably a place to save a bit of compute if we need to optimize later
    list_of_lists = [csr.getrow(i).indices.tolist() for i in range(csr.shape[0])]

    # compute homology
    factored = oat.rust.FactoredBoundaryMatrixDowker(
        dowker_simplices=list_of_lists,
        max_homology_dimension=max_homology_dimension,
    )
    homology = factored.homology()
    del homology["birth simplex"]

    # extract a mapping from number labels to native labels for just the nodes contained in at least one cycle representative
    node_list = list(node_label_sequence)
    node_label_dict = {}
    for cycle in homology["cycle representative"]:
        for simplex in cycle["simplex"]:
            for node in simplex:
                node_label_dict[node] = node_list[node]

    return AssociatedSimplicialComplexHomology(
        max_homology_dimension, homology, node_label_dict
    )


def plot_cycle_representative(cycle, coordinate_dictionary=None, embedding_dimension=3):
    """
    Plots a cycle representative in 2 or 3 dimensional Euclidean space.

    Parameters
    ----------
    cycle : pandas.DataFrame
        A cycle representative obtained from a `hypernetx.AssociatedSimplicialComplexHomology` object.
    coordinate_dictionary : dict, optional
        A dictionary mapping nodes to coordinates. If no dictionary is provided then 
        coordinates are generated automatically using multidimensional scaling (MDS).
        Defaults to None.
    embedding_dimension : int, optional
        The dimension of the embedding, should be 2 or 3. Defaults to 3.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A plotly figure representing the cycle. The `fig.data`, which contains a list 
        of the figure's traces, takes the form::
        
            [vertices, simplex_1, simplex_2, ..]
        
        where `vertices` is a Plotly trace for a 2d or 3d Plotly scatter plot trace, 
        and `simplex_1, simplex_2, ...` are traces representing each simplex in the cycle.

    Notes
    -----
    The user can customize the traces in `fig.data`. For example, if `simplex_1` is a 
    triangle, it will be rendered as a filled polygon trace.

    Examples
    --------
    >>> # Assuming you have a cycle representative from homology computation
    >>> fig = plot_cycle_representative(cycle_rep, embedding_dimension=2)
    >>> fig.show()
    """

    #   EXTRACT A LIST OF EDGES WHERE THE CYCLE TAKES NONZERO COEFFICIENTS
    #   ---------------------------------------------------------------------

    simplices = cycle["simplex"].tolist()

    #   GENERATE MDS COORDINATES FOR EACH VERTEX, BASED ON "HOP DISTANCE" WITHIN
    #   THE GRAPH COMPOSED OF EDGES THAT ARE INCIDENT TO TRIANGLES IN THE CYCLE
    #   ------------------------------------------------------------------------

    if coordinate_dictionary is None:
        coo = oat.plot.hop_mds_from_simplices(
            simplices, dimension=embedding_dimension
        )  # coo stands for "coordinate oracle"
    else:
        coo = coordinate_dictionary
    x = [pt[0] for pt in coo.values()]
    y = [pt[1] for pt in coo.values()]
    if embedding_dimension == 3:
        z = [pt[2] for pt in coo.values()]

    #   GENERATE A TRACE FOR THE POINT CLOUD
    #   ------------------------------------

    data = []

    if embedding_dimension == 3:
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,  # x, y, z coordinates
            mode="markers+text",  # indicates we want some text to appear next to each marker
            text=list(coo.keys()),  # the text we want to appear next to each point
            textposition="top center",  # where we want the text positioned, relative to the marker
            name="Nodes",
        )
        data.append(trace)
    elif embedding_dimension == 2:
        trace = go.Scatter(
            x=x,
            y=y,  # x, y coordinates
            mode="markers+text",  # indicates we want some text to appear next to each marker
            text=list(coo.keys()),  # the text we want to appear next to each point
            textposition="top center",  # where we want the text positioned, relative to the marker
            name="Nodes",
        )
        data.append(trace)

    #   GENERATE A TRACE FOR EACH EDGE
    #   ----------------------------------

    for simplex in simplices:
        if len(simplex) == 2:
            trace = (
                oat.plot.edge__trace3d(edge=simplex, coo=coo)
                if embedding_dimension == 3
                else oat.plot.edge__trace2d(edge=simplex, coo=coo)
            )
            trace.update(
                line=dict(color="red"),  # let's color the edge red
                opacity=0.5,
                showlegend=True,  # indicate we want this simplex to appear in the legend
                name=f"Simplex {simplex}",  # label in the legend entry
                text=f"Vertices: {simplex}",  # text we want to appear when hovering the cursor over the simple
            )
            data.append(trace)
        elif len(simplex) == 3:
            trace = (
                oat.plot.triangle__trace3d(triangle=simplex, coo=coo)
                if embedding_dimension == 3
                else oat.plot.triangle__trace2d(triangle=simplex, coo=coo)
            )
            trace.update(
                opacity=0.5,
                showlegend=True,  # indicate we want this simplex to appear in the legend
                name=f"Simplex {simplex}",  # label in the legend entry
                text=f"Vertices: {simplex}",  # text we want to appear when hovering the cursor over the simple
            )
            data.append(trace)

    #   ADJUST THE PLOT LAYOUT
    #   ----------------------

    fig = go.Figure(data)
    fig.update_layout(
        title="Hover cursor over a simplex to show its list of vertices<br>Click legend entries to toggle on/off",
        width=1000,
        height=1000,
        scene=dict(
            aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1),  # controls zoom
            xaxis=dict(
                range=[-1, 1],
            ),  # x axis limits
            yaxis=dict(
                range=[-1, 1],
            ),  # y axis limits
            zaxis=dict(
                range=[-1, 1],
            ),  # z axis limits
        ),
    )

    return fig
