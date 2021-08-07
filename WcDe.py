# -*- coding: utf-8 -*-

'''

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''

# Importing Libraries
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AHC   

def cluster_word_vectors(
    word_vectors, 
    clustering_algorithm, 
    n_clusters=1500, 
    distance_threshold=None,
    linkage="ward",
    random_state=0,
    **clustering_kwargs
    ):
    '''
    Clusters word vectors

    Parameters
    ----------
    word_vectors : pandas.Series
        Pandas Series with words as the index and vector (numpy array) as the 
        corresponding value.        
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
    
    if type(word_vectors) == pd.core.series.Series:
        word_vectors = list(word_vectors.tolist())

    if clustering_algorithm == "kmeans":
        clustering_model = KMeans(
                                    n_clusters          = n_clusters, 
                                    random_state        = random_state,
                                    **clustering_kwargs
                                 )

    elif clustering_algorithm == "ahc" :        
        clustering_model = AHC(
                                    n_clusters          = n_clusters,
                                    distance_threshold  = distance_threshold,
                                    linkage             = linkage,
                                    compute_full_tree   = True if n_clusters is None else False,
                                    **clustering_kwargs
                                )  
    else:
        raise Exception('''Argument clustering_algorithm accepts "kmeans" or "ahc" only.''')        
        
    clustering_model = clustering_model.fit(word_vectors)

    labels = clustering_model.labels_
    
    if len(set(labels)) == 1:            
        warnings.warn("Based on the parameters provided, only 1 word cluster was found.")

    return labels


def get_document_vectors(    
    tokenized_texts, 
    words,
    cluster_labels, 
    weight_function="cfidf", 
    training_tokenized_texts=None,
    normalize=True
    ):    

    '''
    Computes the document vectors given word clusters and weight function

    Parameters
    ----------
    tokenized_texts : list of lists of str
        A list of texts where each text is further a list of tokens.        
    words : list
        The words that were clustered. 
    cluster_labels : list
        For each word, this list gives the label of the cluster it belongs to.
    weight_function : str, optional
        Provides the default weighting scheme to be used. The available 
        options are "cfidf" and "tfidf_sum".
        The default is "cfidf".
    training_tokenized_texts : list of list of str, optional
        The training texts are used to calculate the document frequencies.       
        The default is None.
    normalize : bool, optional
        If True, then the document vectors will be length normalized (L2).
        The default is True.

    Returns
    -------
    labels : list of lists
        A list of document vector for each documents in the tokenized_texts

    '''

     
    if training_tokenized_texts == None:
        training_tokenized_texts = tokenized_texts

    if type(tokenized_texts) == pd.core.series.Series:
        tokenized_texts = list(tokenized_texts.tolist())
        
    if type(training_tokenized_texts) == pd.core.series.Series:
        training_tokenized_texts = list(training_tokenized_texts.tolist())
        
    # Calculate cluster document frequency
    '''
        Required for CF-iDF cluster weight computation
        
        cf(i,j)  = count of every occurrence of any terms of cluster i that are present in document j
        cdf(i)   = cluster document frequency of cluster i
                 = number of document where any term from cluster i appeared in
    '''
    # Total number of texts (used in tfidf and cfidf weight calculations)
    N = len(training_tokenized_texts) 
    vector_size = len(set(cluster_labels))
    
    # Prepare document frequencies based on the [training] data
    df = {word:[] for word in words}
    cdf = {cl:[] for cl in range(vector_size)}
    word_clusters = {word: cluster_label for word, cluster_label in zip(words, cluster_labels)}

    for idx, tokenized_text in enumerate(training_tokenized_texts):
        for token in set(tokenized_text):
            if token in words:
                if idx not in df[token]:
                    df[token].append(idx)
                if idx not in cdf[word_clusters[token]]:
                    cdf[word_clusters[token]].append(idx)
    
    df = {token:len(df[token]) for token in df}
    cdf = {cluster_label:len(cdf[cluster_label]) for cluster_label in cdf}

    if weight_function == "cfidf":
        #  CF-IDF Prep
        cdf_vector = np.array([cdf[cl] for cl in range(vector_size)]) 
        icdf_vector = np.log(N/cdf_vector + 0.01)
        
    elif weight_function == "tfidf_sum":
        # TF-iDF SUM Prep
        idf = {token:np.log(N/df[token]) for token in df}
    
    wcde_doc_vectors = []
    
    cf_all = []
    for idx, tokenized_text in enumerate(tokenized_texts):
        
        # Default
        cluster_weights = [0]*vector_size 
        
        if len(tokenized_text) > 0:
            # Get terms and frequencies from tokenized text
            
            if weight_function == "cfidf":            
                '''
                # Get cluster frequency - 
                cf(i,j) = count of every occurrence of any terms of cluster i that are present in document j
                '''
                cf_vector = [0]*vector_size 
                for token in tokenized_text:
                    if token in words:
                        cf_vector[word_clusters[token]] += 1                
                cf_vector = np.array(cf_vector)
                cf_all.append(cf_vector)
                
                cluster_weights = cf_vector * icdf_vector
                    
            elif weight_function == "tfidf_sum":  
                for token in set(tokenized_text):
                    if token in words:
                        cluster_weights[word_clusters[token]] += tokenized_text.count(token)*idf[token] 
                        
                cluster_weights = np.array(cluster_weights)
                
            if bool(normalize): 
                cluster_weights = cluster_weights/np.sqrt(np.sum(np.square(cluster_weights)))
            
        wcde_doc_vectors.append(list(np.asarray(cluster_weights)))
    
    return wcde_doc_vectors