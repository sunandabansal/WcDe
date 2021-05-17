# -*- coding: utf-8 -*-

'''
Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''

# Prep

# Imports

import os
import csv
import sys
import json
import time
import scipy
import socket
import pickle
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
from importlib import reload

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.pairwise import paired_distances as sklearn_paired_distances

from gensim.models import Word2Vec as  Gensim_Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Plotting
import seaborn as sns
import matplotlib.pylab as plt

from importlib import reload

### Variables

"""### Functions"""


def cluster_members(cluster_num):
    global inverted_index
    list_of_occurrences = inverted_index[inverted_index.label == cluster_num]\
                        .sort_values("total_tf",ascending=False)["word"]
    words = []
    for word in list_of_occurrences:
        if word not in words:
            words.append(word)
    return words

def print_clusters(cluster_nums, print_limit=20):
    if type(cluster_nums) == int:
        cluster_nums = [cluster_nums]
    tabled_results = []
    for cn in cluster_nums:
        print(cn)
        print("-"*len(str(cn)))
        print(", ".join(cluster_members(cn)[:print_limit]))
        print()

import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AHC    

def cluster_word_vectors(
    word_vectors, 
    clustering_algorithm, 
    n_clusters=1500, 
    distance_threshold=None,
    linkage="ward",
    **clustering_kwargs
    ):
    '''
    Clusters word vectors

    Parameters
    ----------
    word_vectors : 2D numpy.array or pandas.Series
        A 2D array of shape (n_words, word_vector_size) or a Pandas Series with 
        words as the index and vector (numpy array) as the corresponding value.        
    clustering_algorithm : str
        Specifies which clustering algorithm to use. Accepts "kmeans" for K-Means
        or "ahc" for Agglomerative Hierarchical Clustering (AHC). 
    n_clusters : int, optional
        The number of clusters to cluster the word vectors into. The default is 1500.
    distance_threshold : float, optional
        For Agglomerative Hierarchical Clustering (AHC), the clusters with linkage 
        distance smaller than this value will be merged. If a value is declared 
        for this parameter then `n_clusters` must be set to None.        
        The default is None.
    linkage : str, optional
        The linkage critera for Agglomerative Hierarchical Clustering (AHC).
        The default is None.
    **clustering_kwargs : TYPE
        Additional keyword arguments are passed to respective clustering functions
        based on the value `clustering_algorithm`. For refer to the documentation
        of sklearn library for KMeans and AgglomerativeClustering clustering
        algorithms.

    Returns
    -------
    labels : list
        A list of cluster labels for each of the word vectors.

    '''
    
    if type(word_vectors) == pd.Series:
        word_vectors = list(word_vectors.tolist())
       
    if clustering_algorithm == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, **clustering_kwargs)

    elif clustering_algorithm == "ahc" :
        
        if n_clusters is not None: compute_full_tree=False
        
        clustering_model = AHC(
                                    n_clusters          = n_clusters,
                                    distance_threshold  = distance_threshold,
                                    linkage             = linkage,
                                    compute_full_tree   = True if n_clusters is None else False,
                                    **clustering_kwargs
                                )          
        
    clustering_model = clustering_model.fit(word_vectors)

    labels = clustering_model.labels_
    
    if len(set(labels)) == 1:            
        warnings.warn("Based on the parameters provided, only 1 word cluster was found.")

    return labels

def get_document_vectors(tokenized_texts, **kwargs):
    return [
                get_document_vector(tokenized_text, **kwargs) 
                for tokenized_texts in tokenized_texts
           ]

def get_document_vector(
    data, 
    word_vectors, 
    cluster_labels, 
    weight_function="cfidf", 
    normalize=True
    ):
    
    wec_df = word_vectors.to_frame()

    # Name of index
    wec_df.index.name = "word"

    # Add Labels
    wec_df["label"] = labels

    # Get cluster center for each word
    cluster_centers_df = pd.DataFrame(cluster_centers).apply(np.asarray, axis=1).to_frame(name="cluster_center")

    wec_df = wec_df.join(cluster_centers_df, on="label")  
    
    N = len(data)
    if cluster_function == "cfidf" or cluster_function == "tfidf_sum":
        # Calculate cluster document frequency
        '''
        cdf(i,j) = cluster document frquency of cluster i in document j
                 = number of times any term from cluster i appeared in document j
        Required for Qimin's cluster weight computation
        '''
    
        # Inverted Index - word, document, term frequency, cluster label
        inverted_index = pd.DataFrame()
        for idx, text_unit in data["tokenized_text"].iteritems():
            if text_unit:
    
                # Get terms and frequencies from tokenized text
                tf = pd.Series(text_unit, name="term_freq").value_counts()
    
                # Filter Word Cluster index (wec_df) to contain only words from the Text, reset to not lose words (index)
                wec_text_df = wec_df[["label"]].join(tf).dropna(subset=["term_freq"]).reset_index()
    
                wec_text_df["doc_id"] = idx
    
                inverted_index = inverted_index.append(wec_text_df)
    
        inverted_index = inverted_index.rename(columns={"index":"word"})

        if cluster_function == "cfidf":
            # Get number of documents for each cluster - count unique doc_ids
            cdf = inverted_index.groupby(["label"])["doc_id"].nunique().rename("df")
    
            # Get cluster idf values
            icdf_vector = np.log(N/cdf + 0.01)
            
        elif cluster_function == "fraj" or cluster_function == "tfidf_sum":
            # Get term document frequency
            doc_freq = inverted_index.groupby(["word"])["doc_id"].nunique().rename("df")
            
            # Inverse document frequency
            wec_df["idf"] = np.log(N/doc_freq)
    
    list_of_text_embeddings = []
    for i, text_unit in enumerate(data["tokenized_text"]):
        if text_unit:
    
            # Get terms and frequencies from tokenized text
            tf = pd.Series(text_unit, name="term_freq").value_counts()
            
            # Filter Word Cluster index (wec_df) to contain only words from the Text, reset to not lose words (index)
            wec_text_df = wec_df.join(tf).dropna(subset=["term_freq"]).reset_index()
            
            if cluster_function == "cfidf":            
                '''
                # Get cluster frequency - 
                cf(i,j) = count of every occurrence of any terms of cluster i that are present in document j
                '''
                cf_vector = wec_text_df.groupby(["label"])["term_freq"].apply(np.sum)
                
                cluster_weights = cf_vector * icdf_vector[cf_vector.index]
                
                if bool(wcde_vector_normalize): 
                    cluster_weights = cluster_weights/np.sqrt(np.sum(np.square(cluster_weights)))
                    
            elif cluster_function == "tfidf_sum":            
                
                wec_text_df["tfidf"] = wec_text_df.term_freq * wec_text_df.idf
                
                cluster_weights = wec_text_df.groupby(["label"])["tfidf"].apply(np.sum)
                
                if bool(wcde_vector_normalize): 
                    cluster_weights = cluster_weights/np.sqrt(np.sum(np.square(cluster_weights)))
                    
            elif cluster_function == "CDBWA":
                # For each cluster, indexed by label - list of words, list of embeddings and list of cluster_centers 
                clusters = wec_text_df.groupby(["label"]).agg(list)
    
                # Get Weight Sum
                # cluster_label : list of distances for each participating word
                cluster_weights = clusters[["embedding","cluster_center"]].apply(distance_weighted_sum, axis=1).apply(np.sum)                
                
                # Average
                cluster_weights = cluster_weights/len(wec_text_df)
                
            elif cluster_function == "CDWA":
                # For each cluster, indexed by label - list of words, list of embeddings and list of cluster_centers 
                clusters = wec_text_df.groupby(["label"]).agg(list)
    
                # Get Weight Sum
                # cluster_label : list of distances for each participating word
                cluster_weights = clusters[["embedding","cluster_center","term_freq"]].apply(distance_weighted_sum, axis=1).apply(np.sum)                 
                
                # Average
                cluster_weights = cluster_weights/len(wec_text_df)
                
            # In case there are clusters not seen in the sentence
            cluster_labels = wec_df.label.sort_values().unique()
    
            # Re-indexing to include clusters unseen clusters
            cluster_weights = cluster_weights.reindex(cluster_labels, fill_value=0) 
            
            list_of_text_embeddings.append(np.asarray(cluster_weights))
        else:
            # The text is empty
            zero_vector = np.zeros(len(wec_df.label.sort_values().unique()))
            list_of_text_embeddings.append(zero_vector)
            
    return list_of_text_embeddings