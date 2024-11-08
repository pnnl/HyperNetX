import hypernetx as hnx
import networkx as nx
import numpy as np
import pandas as pd

from hypernetx.algorithms.hedge2vec import Hedge2Vec


test_disconn_hg = hnx.Hypergraph({'A': [1], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3], 'F': [4]})
test_conn_hg =  hnx.Hypergraph({'A': [1], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3]})

test_bipartite_graph = hnx.Hypergraph({'A': [1], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3]}).bipartite() # same as above

test_bipartite_graph_weighted = hnx.Hypergraph({'A': [1], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3]}).bipartite()
for i, edge in enumerate(test_bipartite_graph_weighted.edges()):
    test_bipartite_graph_weighted.edges[edge]['weight']=i//2+1

def test_check_whitespace():
    h2v = Hedge2Vec()
    
    # checking exception not raised when no whitespace
    h2v.check_whitespace(test_conn_hg)
    
    # checking exception raised when whitespace in edges
    H1 = hnx.Hypergraph({'A ': [1], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3]})
    try:
        h2v.check_whitespace(H1)
        raise Exception('Hedge2Vec whitespace check failed')
    except:
        pass
    
    # checking exception raised when whitespace in nodes
    H2 = hnx.Hypergraph({'A ': [' 1'], 'B': [2], 'C': [2], 'D': [3], 'E': [1, 2, 3]})
    try:
        h2v.check_whitespace(H2)
        raise Exception('Hedge2Vec whitespace check failed')
    except:
        pass
    
def test_connectedness():
    h2v = Hedge2Vec()
    
    # checking passes when connected
    h2v.connectedness(test_conn_hg)
    
    # checking exception raised when not connected
    try:
        h2v.connectedness(test_disconn_hg)
        raise Exception('Hedge2Vec connectedness failed')
    except:
        pass
    
def test_train_node2vec():
    h2v = Hedge2Vec(weighted=False)
    m1 = h2v.train_node2vec(test_bipartite_graph)
    h2v = Hedge2Vec(weighted=True)
    m2 = h2v.train_node2vec(test_bipartite_graph_weighted)
    assert m1.wv.vectors.shape[1] == h2v.vector_size
    assert m1.wv.vectors.shape == m2.wv.vectors.shape

def test_get_embedding_dict():
    h2v = Hedge2Vec()
    m = h2v.train_node2vec(test_bipartite_graph)
    assert len(h2v.get_embedding_dict(m)) == len(list(test_bipartite_graph.nodes()))

def test_get_unweighted_line_graphs():
    h2v = Hedge2Vec(weighted=False)
    G1_nodes, G1_edges = h2v.get_unweighted_line_graphs(test_conn_hg)
    assert len(list(G1_nodes.nodes())) == len(list(test_conn_hg.nodes()))
    assert len(list(G1_edges.nodes())) == len(list(test_conn_hg.edges()))

def test_get_weighted_line_graphs():
    h2v = Hedge2Vec(weighted=True)
    G1_nodes, G1_edges = h2v.get_weighted_line_graphs(hnx.Hypergraph.from_bipartite(test_bipartite_graph_weighted))
    assert len(list(G1_nodes.nodes())) == len(list(test_conn_hg.nodes()))
    assert len(list(G1_edges.nodes())) == len(list(test_conn_hg.edges()))

def test_fit_transform():
    # note - not testing fit separately, just here

    # connected HG input
    h2v = Hedge2Vec(weighted=False, vectorization='edges')
    edges_out = h2v.fit_transform(test_conn_hg)
    h2v = Hedge2Vec(weighted=True, vectorization='nodes')
    nodes_out = h2v.fit_transform(hnx.Hypergraph.from_bipartite(test_bipartite_graph_weighted))
    assert edges_out.shape[1] == nodes_out.shape[1]

def test_check_fitted():
    h2v = Hedge2Vec()
    try:
        h2v.check_fitted()
        raise Exception('Hedge2Vec check fitted failed on unfitted model')
    except:
        pass        
    h2v.fit(test_conn_hg)
    h2v.check_fitted()

def test_get_dist_matrix():
    h2v = Hedge2Vec()
    h2v.fit(test_conn_hg)
    h2v.dist_metric = 'cosine'
    D = h2v.get_dist_matrix()
    # checking symmetric with a tolerance
    assert np.all(np.abs(D-D.T) < 1e-08)

def test_dim_red_embeddings():
    h2v = Hedge2Vec()
    h2v.fit(test_conn_hg)
    umap_x, umap_y = h2v.dim_red_embeddings(dim_red='umap', dist_metric='cosine')
    mds_x, mds_y = h2v.dim_red_embeddings(dim_red='mds', dist_metric='cosine')
    assert len(umap_x) == len(umap_y) == len(mds_x) == len(mds_y)

# def test_plot_embeddings():
#     h2v = Hedge2Vec()
#     h2v.fit(test_conn_hg)
#     emb_2d = h2v.dim_red_embeddings(dim_red='mds', dist_metric='cosine')
#     cluster_labels={i:i//2 for i in range(len(emb_2d[0]))}
#     print(cluster_labels)
#     h2v.plot_embeddings(emb_2d=emb_2d)
#     h2v.plot_embeddings(emb_2d=emb_2d, cluster_labels={list(test_conn_hg.edges)[i]:i//2 for i in range(len(emb_2d[0]))})

def test_cluster_embeddings():
    h2v = Hedge2Vec()
    h2v.fit(test_conn_hg)
    embs = h2v.cluster_embeddings(dist_metric='cosine')
    assert len(embs) == len(test_conn_hg.edges)

