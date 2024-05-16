
.. _hypconstructors:

=======================
Hypergraph Constructors
=======================

An hnx.Hypergraph H = (V,E) references a pair of disjoint sets:
V = nodes (vertices) and E = (hyper)edges.

HNX allows for multi-edges by distinguishing edges by
their identifiers instead of their contents. For example, if
V = {1,2,3} and E = {e1,e2,e3},
where e1 = {1,2}, e2 = {1,2}, and e3 = {1,2,3},
the edges e1 and e2 contain the same set of nodes and yet
are distinct and are distinguishable within H = (V,E).

HNX provides methods to easily store and
access additional metadata such as cell, edge, and node weights.
Metadata associated with (edge,node) incidences
are referenced as **cell_properties**.
Metadata associated with a single edge or node is referenced
as its **properties**.

The fundamental object needed to create a hypergraph is a **setsystem**. The
setsystem defines the many-to-many relationships between edges and nodes in
the hypergraph. Cell properties for the incidence pairs can be defined within
the setsystem or in a separate pandas.Dataframe or dict.
Edge and node properties are defined with a pandas.DataFrame or dict.

SetSystems
----------
There are five types of setsystems currently accepted by the library.

1.  **iterable of iterables** : Barebones hypergraph, which uses Pandas default
    indexing to generate hyperedge ids. Elements must be hashable.: ::

    >>> list_of_lists = [['book','candle','cat'],['book','coffee cup'],['coffee cup','radio']]
    >>> H = Hypergraph(list_of_lists)

2.  **dictionary of iterables** : The most basic way to express many-to-many
    relationships providing edge ids. The elements of the iterables must be
    hashable): ::

    >>> scenes_dictionary = {
    >>> 	0: ('FN', 'TH'),
    >>> 	1: ('TH', 'JV'),
    >>> 	2: ('BM', 'FN', 'JA'),
    >>> 	3: ('JV', 'JU', 'CH', 'BM'),
    >>> 	4: ('JU', 'CH', 'BR', 'CN', 'CC', 'JV', 'BM'),
    >>> 	5: ('TH', 'GP'),
    >>> 	6: ('GP', 'MP'),
    >>> 	7: ('MA', 'GP'),
    >>> 	8: ('FN', 'TH')}
    >>> H = hnx.Hypergraph(scenes_dictionary)

3.  **dictionary of dictionaries**  : allows cell properties to be assigned
    to a specific (edge, node) incidence. This is particularly useful when
    there are variable length dictionaries assigned to each pair: ::

    >>> nested_dictionary =  {
    >>> 	0: {'FN':{'time':'early', 'weight': 7}, 'TH':{'time':'late'}},
    >>> 	1: {'TH':{'subject':'war'}, 'JV':{'observed_by':'someone'}},
    >>> 	2: {'BM':{}, 'FN':{}, 'JA':{'role':'policeman'}},
    >>> 	3: {'JV':{'was_carrying':'stick'}, 'JU':{}, 'CH':{}, 'BM':{'state':'intoxicated', 'color':'pinkish'}},
    >>> 	4: {'JU':{'weight':15}, 'CH':{}, 'BR':{'state':'worried'}, 'CN':{}, 'CC':{}, 'JV':{}, 'BM':{}},
    >>> 	5: {'TH':{}, 'GP':{}},
    >>> 	6: {'GP':{}, 'MP':{}},
    >>> 	7: {'MA':{}, 'GP':{'accompanied_by':'dog', 'weight':15, 'was_singing': 'Frère Jacques'}}}
    >>> H = hnx.Hypergraph(nested_dictionary)

4.  **pandas.DataFrame** For large datasets and for datasets with cell
    properties it is most efficient to construct a hypergraph directly from
    a pandas.DataFrame. Incidence pairs are in the first two columns.
    Cell properties shared by all incidence pairs can be placed in their own
    column of the dataframe. Variable length dictionaries of cell properties
    particular to only some of the incidence pairs may be placed in a single
    column of the dataframe. Representing the data above as a dataframe df:

    +-----------+-----------+-----------+-----------------------------------+
    |   col1    |   col2    |   w       |  col3                             |
    +-----------+-----------+-----------+-----------------------------------+
    |   e1      |   1       |   0.5     | {'name':'related_to'}             |
    +-----------+-----------+-----------+-----------------------------------+
    |   e1      |   2       |   0.1     | {"name":"related_to",             |
    |           |           |           |  "startdate":"05.13.2020"}        |
    +-----------+-----------+-----------+-----------------------------------+
    |   e2      |   1       |   0.52    | {"name":"owned_by"}               |
    +-----------+-----------+-----------+-----------------------------------+

    The first row of the dataframe is used to reference each column. ::

    >>> import pandas as pd
    >>> d = {'col1': ['e1', 'e1', 'e2'],
    >>>      'col2': [1, 2, 1],
    >>>      'w': [0.5, 0.1, 0.52],
    >>>      'col3':[{'name': 'related_to'}, {'name': 'related_to', 'startdate':'05.13.2020'}, {'name': 'owned_by'}]}
    >>> df = pd.DataFrame(d)
    >>> H = hnx.Hypergraph(df, edge_col="col1", node_col="col2",
    >>>                    cell_weight_col="w", misc_cell_properties_col="col3")

5.  **numpy.ndarray** For homogeneous datasets given in a *n x 2* ndarray a
    pandas dataframe is generated and column names are added from the
    edge_col and node_col arguments. Cell properties containing multiple data
    types are added with a separate dataframe or dict and passed through the
    cell_properties keyword. ::

    >>> import bumpy as np
    >>> np_array = np.array([['A','a'],['A','b'],['A','c'],['B','a'],['B','d'],['C','c'],['C','d']])
    >>> H = hnx.Hypergraph(np_array)


Edge and Node Properties
------------------------
Properties specific to edges and/or node can be passed through the
keywords: **edge_properties, node_properties, properties**.
Properties may be passed as dataframes or dicts.
The first column or index of the dataframe or keys of the dict keys
correspond to the edge and/or node identifiers.
If properties are specific to an id, they may be stored in a single
object and passed to the **properties** keyword. For example:

+-----------+-----------+---------------------------------------+
|   id      |   weight  |   properties                          |
+-----------+-----------+---------------------------------------+
|   e1      |   5.0     |   {'type':'event'}                    |
+-----------+-----------+---------------------------------------+
|   e2      |   0.52    |   {"name":"owned_by"}                 |
+-----------+-----------+---------------------------------------+
|   ...     |   ...     |   {...}                               |
+-----------+-----------+---------------------------------------+
|   1       |   1.2     |   {'color':'red'}                     |
+-----------+-----------+---------------------------------------+
|   2       |   .003    |   {'name':'Fido','color':'brown'}     |
+-----------+-----------+---------------------------------------+
|   3       |   1.0     |    {}                                 |
+-----------+-----------+---------------------------------------+

A properties dictionary should have the format: ::

    dp = {id1 : {prop1:val1, prop2,val2,...}, id2 : ... }

A properties dataframe may be used for nodes and edges sharing ids
but differing in cell properties by adding a level index using 0
for edges and 1 for nodes:

+-----------+-----------+-----------+---------------------------+
|  level    |   id      |   weight  |       properties          |
+-----------+-----------+-----------+---------------------------+
|   0       |   e1      |   5.0     |   {'type':'event'}        |
+-----------+-----------+-----------+---------------------------+
|   0       |   e2      |    0.52   |   {"name":"owned_by"}     |
+-----------+-----------+-----------+---------------------------+
|   ...     |   ...     |    ...    |          {...}            |
+-----------+-----------+-----------+---------------------------+
|   1       |   1.2     |   {'color':'red'}                     |
+-----------+-----------+-----------+---------------------------+
|   2       |   .003    |   {'name':'Fido','color':'brown'}     |
+-----------+-----------+-----------+---------------------------+
|   ...     |   ...     |    ...    |          {...}            |
+-----------+-----------+-----------+---------------------------+



Weights
-------
The default key for cell and object weights is "weight". The default value
is 1. Weights may be assigned and/or a new default prescribed in the
constructor using **cell_weight_col** and **cell_weights** for incidence pairs,
and using **edge_weight_prop, node_weight_prop, weight_prop,
default_edge_weight,** and **default_node_weight** for node and edge weights.