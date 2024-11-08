import os
import umap

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

from collections import Counter
from gensim.models import Word2Vec
import hypernetx as hnx
from matplotlib import pyplot as plt
from networkx.algorithms import bipartite
from pecanpy import pecanpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import HDBSCAN
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

class HypergraphVectorizer(BaseEstimator, TransformerMixin):
    """
        Initializes the Hedge2Vec Object

        Parameters
        ----------
        weighted : boolean, default: False
            whether or not we are embedding a weighted hypergraph, if so this assumes the weights are in the data in a form HypernetX expects

        vectorization : string, default: 'edges'
            'edges' or 'nodes' based on if the final embeddings are hyperedge or hypernode embeddings
    """

    def __init__(self, weighted=False, vectorization='edges'):
        self.weighted = weighted
        self.vectorization = vectorization
        self._train_matrix = None
        self._train_names = None
   
    def fit(self, X, y=None):
        raise Exception('This must be overwritten to use')

    def fit_transform(self, X, y=None):
        """
            Takes in any hypernetx accepted input and trains Hedge2Vec on the associated hypergraph and returns the learned embedding

            Parameters
            ----------
            X : HypernetX Hypergraph

            Returns
            -------
            numpy array of learned embedding vectors

        """
        self.fit(X, y=y)
        return self._train_matrix
        
    def check_fitted(self):
        """
            Checks that model has been fitted and raises an exception otherwise

            Parameters
            ----------


            Returns
            -------

        """
        if self._train_matrix is None:
            raise Exception('Model is not fitted yet - fit model first and then rerun')
            
    def get_dist_matrix(self):
        """
            Checks that model has been fitted and raises an exception otherwise

            Parameters
            ----------
            dist_metric : str, default: 'cosine'
                distance metric used for dimensionality reduction or clustering


            Returns
            -------
            numpy array of distance matrix

        """
        self.check_fitted()
        if self.dist_metric.lower() == 'cosine':
            dist_mat = 1 - cosine_similarity(self._train_matrix)
        else:
            raise Exception(f"Distance metric {self.dist_metric} not implemented for {self.dim_red} - try any of ['cosine']")
        return dist_mat
    
    def dim_red_embeddings(self, dim_red='mds', dist_metric='cosine'):
        """
            Checks that model has been fitted and raises an exception otherwise

            Parameters
            ----------
            dim_red : str, default: 'mds'
                dimensionality reduction algorithm, mds or umap currently implemented
            
            dist_metric : str, default: 'cosine'
                distance metric used for dimensionality reduction


            Returns
            -------
            x,y : lists of x and y coordinates respectively for plotting

        """
        self.check_fitted()
        self.dim_red = dim_red
        self.dist_metric = dist_metric
        if dim_red.lower() == 'umap':
            reducer = umap.UMAP(metric=dist_metric)
            plotting_arr = reducer.fit_transform(self._train_matrix)
        
        elif dim_red.lower() == 'mds':
            dist_mat = self.get_dist_matrix()
            mds = MDS(dissimilarity='precomputed')
            plotting_arr = mds.fit_transform(dist_mat)

        else:
            raise Exception(f"Dimensionality reduction {dim_red} not implemented - try any of ['mds', 'umap']")
            
        return list(zip(*plotting_arr.tolist()))


    def plot_embeddings(self, emb_2d, cluster_labels=[]):
        """
            Created 2D Multi-Dimensional Scaling Plots from Cosine Similarity of distance matrix
            Note: Model must be fitted first

            Parameters
            ----------
            emb_2d: tuple (x,y) of coordinates of 2D embedding
            
            cluster_labels: dict, default: []
                a dictionary of cluster label lookups

            Returns
            -------

        """
        self.check_fitted()
        x,y = emb_2d

        if len(cluster_labels) == 0:
            # creates dataframe with plotting data to be passed to plotly and plots
            plotting_df = pd.DataFrame([{'x': xx, 'y': yy, 
                                         self.vectorization[:-1]: h} 
                                        for (xx,yy,h) in zip(x,y, self._train_names)])
        else:
            plotting_df = pd.DataFrame([{'x': xx, 'y': yy, 
                             self.vectorization[:-1]: h,
                             'cluster_label': str(cluster_labels[h])} 
                            for (xx,yy,h) in zip(x,y, self._train_names)])
            
        fig = px.scatter(plotting_df, x="x", y="y", 
                         hover_data=[self.vectorization[:-1]],
                         color='cluster_label' if len(cluster_labels) > 0 else None,
                         color_discrete_sequence=px.colors.qualitative.Light24,
                         width=1000,
                         height=500,
                         title=f'{self.dim_red} {self.dist_metric} Distance Plot')
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.show()
        
    def cluster_embeddings(self, dist_metric='cosine', min_cluster_size=2):
        """
            Clusters embedding vectors using HDBSCAN and plots color-coded scatter plot
            Note: Model must be fitted first

            Parameters
            ----------
            dist_metric : str, default: 'cosine'
                distance metric used for clustering
            
            min_cluster_size: minimum cluster size parameter for HDBSCAN

            Returns
            -------
            dictionary of cluster labels

        """
        
        # checks that the model has been fit and raises an exception if not
        self.check_fitted()
        
        # computes the distance matrix from the cosine similarity of the embedding vectors
        self.dist_metric = dist_metric
        dist_mat = self.get_dist_matrix() 
            
        # clusters the data from the distance matrix
        # TODO: Include other clustering methods
        model = HDBSCAN(min_cluster_size=min_cluster_size,metric='precomputed')
        model.fit(np.array(dist_mat))
        
        # returns dictionary of cluster labels
        return dict(zip(self._train_names, model.labels_))

class Hedge2Vec(HypergraphVectorizer):
    """
        Initializes the Hedge2Vec Object

        Parameters
        ----------
        vector_size : int > 0, default: 64
            size of the embedding vector sizes from each of the line graphs

        walk_length : int > 0, default: 30
            the length of the random walks on the line graph in the node2vec algorithm

        num_walks : int > 0, default: 200
            the number of random walks starting at each node in the node2vec algorithm

        workers : int > 0, default: 1
            number of threads to be spawned for running node2vec including walk generation and word2vec embedding

        window : int > 0, default: 10
            maximum distance between the current and predicted word within a sentence for training the Word2Vec model

        min_count : int > 0, default: 1
            Word2Vec model ignores all words with total frequency lower than this

        sg : int {0, 1}, default: 1
            Training algorithm: 1 for skip-gram; otherwise CBOW

        epochs : int > 0, default: 1
            the number of epochs for training the Word2Vec model
    """
    
    # super init
    def __init__(self, weighted=False, vectorization='edges', centroid_alpha=1.0, vector_size=64, walk_length=30, num_walks=200, workers=1, window=10, min_count=1, sg=1, epochs=1):
        super().__init__(weighted, vectorization)
        self.centroid_alpha = centroid_alpha
        self.vector_size = vector_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs 
    
    def check_whitespace(self, H):
        """
            If hypergraph node or edge names contain whitespace raises an exception

            Parameters
            ----------
            H : hnx.Hypergraph object

            Returns
            -------
        """
        if any([len(str(n).split()) > 1 for n in H.nodes()] + [len(str(e).split()) > 1 for e in H.edges()]):
            raise Exception('Hypergraph node or edge names contain whitespace - please remove whitespace in the names')
        else:
            print('no whitespace issues')
    
    def connectedness(self, H):
        """
            If hypergraph is disconnected raises an exception

            Parameters
            ----------
            H : hnx.Hypergraph object

            Returns
            -------
            HNX Hypergraph
        """
        if not nx.is_connected(H.bipartite()):
            raise Exception('Hypergrah is not connected - please use a connected hypergraph')
        return H

    def train_node2vec(self, graph):
        """
            Trains the node2vec algorithm on a NetworkX graph

            Parameters
            ----------
            graph : NetworkX graph, default: None

            Returns
            -------
            trained Word2Vec model
        """
        # save graph data to temporary file for Node2Vec
        if not self.weighted:
            nx.write_edgelist(graph, 'tmp.edg', delimiter=' ', data=False)
        else:
            nx.write_edgelist(graph, 'tmp.edg', delimiter=' ', data=['weight'])
            
        g = pecanpy.SparseOTF(p=1, q=1, workers=self.workers, verbose=False, extend=True)
        g.read_edg('tmp.edg', weighted=self.weighted, directed=False, delimiter=' ')
        # delete temporary file
        os.remove('tmp.edg') 
        # generate random walks
        walks = g.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length)
        # use random walks to train embeddings
        model = Word2Vec(walks, vector_size=self.vector_size, window=self.window, min_count=self.min_count, sg=self.sg, workers=self.workers, epochs=self.epochs)
        return model
    
    def get_embedding_dict(self, model):
        """
            Gets the token embedding dictionary of a Word2Vec model 

            Parameters
            ----------
            model : Word2Vec model, default: None

            Returns
            -------
            dictionary with keys names of embedded hypergraph feature and values embedding vectors
        """
        return dict(zip(model.wv.index_to_key,model.wv.vectors))
    
    def get_unweighted_line_graphs(self, H):
        """
            Gets the line graph of an unweighted hypergraph (or ignores weights if weighted)

            Parameters
            ----------
            H : HypernetX Hypergraph object, default: None

            Returns
            -------
            L1, L2: NetworkX graphs - Line graphs associated to nodes and edges respectively
        """
        return H.get_linegraph(edges=False), H.get_linegraph(edges=True)
    
    def prob_weight(self, B, u, v, weight='weight'):
        """
            Method for computing weights between two nodes in a weighted bipartite graph

            Parameters
            ----------
            B : NetworkX bipartite graph, default: None
            
            u: Node in B
            
            v: Node in B
            
            weight: str, default: 'weight'
                key of the weight field in B

            Returns
            -------
            dictionary with keys names of embedded hypergraph feature and values embedding vectors
        """
        w = 0
        for nbr in set(B[u]).intersection(set(B[v])):
            w += B.edges[(u, nbr)]['weight']*B.edges[(v, nbr)]['weight']
        return w
    
    def get_weighted_line_graphs(self, H):
        """
            Method for getting weighted line graphs from a weighted hypergraph

            Parameters
            ----------
            H : HypernetX hypergraph object, default: None

            Returns
            -------
            G1, G1 : NetworkX weighted bipartite graphs for nodes and edges respectively
        """
        
        B = H.bipartite(keep_data=False)
        for e, w in H.properties.to_dict()['weight'].items():
            B.edges[e]['weight'] = w
        P_edge = bipartite.generic_weighted_projected_graph(B,bipartite.sets(B)[0], weight_function=self.prob_weight)
        P_node = bipartite.generic_weighted_projected_graph(B,bipartite.sets(B)[1], weight_function=self.prob_weight)
        return P_node, P_edge
    
    def fit(self, X, y=None):
        """
            Takes in any hypernetx accepted input and trains Hedge2Vec on the associated hypergraph

            Parameters
            ----------
            X : HNX Hypergraph, default: None

            Returns
            -------
            self

        """
        self.check_whitespace(X)
        H = self.connectedness(X)
        
        # Gets hyperedge/node line graphs and trains 2 separate node2vec models
        if not self.weighted:
            G_nodes, G_edges = self.get_unweighted_line_graphs(H)
        else:
            G_nodes, G_edges = self.get_weighted_line_graphs(H)
            
        model_nodes = self.train_node2vec(G_nodes) 
        model_edges = self.train_node2vec(G_edges)
        
        inc_dict = {str(k):v for k,v in H.incidence_dict.items()}
        dual_dict = {str(k):v for k,v in H.dual().incidence_dict.items()}
        
        # vectorizes nodes/hyperedges from aggregation of node2vec embeddings
        vectorization_dict = {}
        if self.vectorization == 'nodes':
            print('vectorizing nodes with edge centroids')
            for node, node_v in self.get_embedding_dict(model_nodes).items():
                edge_centr = self.centroid_alpha*np.mean(np.array([self.get_embedding_dict(model_edges)[str(e)] for e in dual_dict[str(node)]]), axis=0)
                vectorization_dict[node] = np.concatenate([node_v, edge_centr])

        elif self.vectorization == 'edges':
            print('vectorizing edges with node centroids')
            for edge, edge_v in self.get_embedding_dict(model_edges).items():
                node_centr = self.centroid_alpha*np.mean(np.array([self.get_embedding_dict(model_nodes)[str(n)] for n in inc_dict[str(edge)]]), axis=0)
                vectorization_dict[edge] = np.concatenate([edge_v, node_centr])
                
        self._train_names = list(vectorization_dict.keys())
        self._train_matrix = np.array(list(vectorization_dict.values()))
        return self