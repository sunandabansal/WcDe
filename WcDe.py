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
    word_vectors, 
    cluster_labels, 
    weight_function="cfidf", 
    normalize=True
    ):
    
    # Prepare dataframe with word vector and cluster labels for each word (index)
    wec_df = word_vectors.to_frame()    
    wec_df["label"] = cluster_labels    # Add Cluster labels to the dataframe    
    wec_df.index.name = "word"          # Name the index

    # Total number of texts (used in cfidf weight calculations)
    N = len(tokenized_texts)           
    
    # Calculate cluster document frequency
    '''
        Required for CF-iDF cluster weight computation
        
        cf(i,j)  = count of every occurrence of any terms of cluster i that are present in document j
        cdf(i,j) = cluster document frquency of cluster i in document j
                 = number of times any term from cluster i appeared in document j

    '''
    # Prepare an index of term occurrence in documents
    # Inverted Index - word, document, term frequency, cluster label
    inverted_index = pd.DataFrame()
    for idx, tokenized_text in enumerate(tokenized_texts):
        if tokenized_text:

            # Get terms and frequencies from tokenized text
            tf = pd.Series(tokenized_text, name="term_freq").value_counts()

            # Filter Word Cluster index (wec_df) to contain only words from the Text, 
            # reset to not lose words (index)
            wec_text_df = wec_df[["label"]].join(tf).dropna(subset=["term_freq"]).reset_index()

            wec_text_df["doc_id"] = idx

            inverted_index = inverted_index.append(wec_text_df)

    inverted_index = inverted_index.rename(columns={"index":"word"})

    if weight_function == "cfidf":
        # Get number of documents for each cluster - count unique doc_ids
        cdf = inverted_index.groupby(["label"])["doc_id"].nunique().rename("df")

        # Get cluster idf values
        icdf_vector = np.log(N/cdf + 0.01)
        
    elif weight_function == "tfidf_sum":
        # Get term document frequency
        doc_freq = inverted_index.groupby(["word"])["doc_id"].nunique().rename("df")
        
        # Inverse document frequency
        wec_df["idf"] = np.log(N/doc_freq)
    
    wcde_doc_vectors = []
    
    for tokenized_text in tokenized_texts:

        if tokenized_text:
    
            # Get terms and frequencies from tokenized text
            tf = pd.Series(tokenized_text, name="term_freq").value_counts()
            
            # Filter Word Cluster index (wec_df) to contain only words from the Text, reset to not lose words (index)
            wec_text_df = wec_df.join(tf).dropna(subset=["term_freq"]).reset_index()
            
            if weight_function == "cfidf":            
                '''
                # Get cluster frequency - 
                cf(i,j) = count of every occurrence of any terms of cluster i that are present in document j
                '''
                cf_vector = wec_text_df.groupby(["label"])["term_freq"].apply(np.sum)
                
                cluster_weights = cf_vector * icdf_vector[cf_vector.index]
                    
            elif weight_function == "tfidf_sum":            
                
                wec_text_df["tfidf"] = wec_text_df.term_freq * wec_text_df.idf
                
                cluster_weights = wec_text_df.groupby(["label"])["tfidf"].apply(np.sum)
                
            if bool(normalize): 
                cluster_weights = cluster_weights/np.sqrt(np.sum(np.square(cluster_weights)))
                
            # Re-indexing to include clusters unseen clusters
            cluster_weights = cluster_weights.reindex(set(cluster_labels), fill_value=0) 
            
            wcde_doc_vectors.append(np.asarray(cluster_weights))
        else:
            # The text is empty
            zero_vector = np.zeros(len(wec_df.label.sort_values().unique()))
            wcde_doc_vectors.append(zero_vector)
            
    return wcde_doc_vectors