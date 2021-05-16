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

from sklearn.cluster import KMeans
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

'''
Not using Abbreviated names of metrics messes up when we rotate the xtick labels for the plots
that contain the metrics on the x-axis
'''
abbrev_metrics = ['Norm. MIS', 'Homo.', 'Compl.', 'V Meas.', 'Fowlkes',
                  'Adj. Rand', 'Purity', 'Accuracy', 'F1 Score', 'Precision', 'Recall']

abbrev_metrics_acronym = ['NMI', 'H', 'C', 'FMI', 'ARI', 'Purity', 'Acc.', 'F1', 'P', 'R']

score_cols = ['Normalized Mutual Information Score', 'Homogeneity Score', 'Completeness Score', 
              'Fowlkes Mallows Score', 'Adjusted Rand Score', 'Purity',
              'Accuracy Score', 'F1 Score']


extra_metrics = [
                     'Calinski-Harabasz Index',
                     'Davies-Bouldin Index',
                     'Silhouette Score'
                ]

metrics_order = score_cols

word_vectors_order = [
            "glove100", 
            "glove300", 
            "word2vec_pretrained", 
            "word2vec_20epochs", 
            "word2vec_50epochs", 
            "word2vec_100epochs", 
            "word2vec_150epochs", 
          ]

word_vectors_name_map = {
    'glove100': "GloVe 100d", 
    'glove300': "GloVe 300d", 
    'word2vec_pretrained': "Word2vec Pre", 
    'wcde_w2v':"Word2vec",
    'word2vec_20epochs':"Word2vec 20 Epochs",
    'word2vec_50epochs':"Word2vec 50 Epochs",
    'word2vec_100epochs':"Word2vec 100 Epochs",
    'word2vec_150epochs':"Word2vec 150 Epochs"
}

word_vectors_order = [word_vectors_name_map[wv] for wv in word_vectors_order]

metric_name_map = {}
for full, acro in zip(score_cols, abbrev_metrics_acronym):
    metric_name_map[full] = acro

wcde_name = "WcDe"
method_name_map = {
                    'weavg':"WE_AVG", 
                    'doc2vec':"Doc2Vec", 
                    'wcde':wcde_name, 
                    'vsm':"VSM", 
                    'lda':"LDA", 
                    'lsi':"LSA", 
                    'SIF':"WE_SIF", 
                    'wetfidf':"WE_TFIDF"
                  }
methods_order = ["WE_AVG", "WE_SIF", "WE_TFIDF", "LDA", "VSM", "LSA", "Doc2Vec", wcde_name]

dataset_name_map = {
                        "bbc":"BBC",
                        "bbcsport":"BBC Sport"                        
                    }


"""### Functions"""

def Sentenced_Tokenizer(text, removeStopwords=False, stem=False):
    sentences = nltk.tokenize.sent_tokenize(text.lower())

    processed_sentences = []
    for sentence in sentences:        
        processed_sentences.append(NLTK_Text_Tokenizer(sentence, removeStopwords=removeStopwords, stem=stem))

    return processed_sentences

# Flatten list
def flatten(l,unique=False):
    flattened_list = []
    for item in l:
        if type(item) is str:
            flattened_list.append(item)
            continue
        try:   
            if iter(item):
                flattened_list.extend(flatten(item))
        except:
            flattened_list.append(item)
    if unique:
        return list(set(flattened_list))     
    else:        
        return flattened_list

def regex_tokenizer(doc, removeStopwords=True):
    """
    Input  : document list
    Purpose: preprocess text (tokenize, removing stopwords)
    Output : preprocessed text
    """
    # clean and tokenize document string
    global tokenizer, en_stop

    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    if removeStopwords:
        tokens = [token for token in tokens if not token in en_stop]

    return tokens


# def load_tokenized_dataset(args, tokenizer, **tokenizer_kwargs):
#     '''
#     Parameters
#     ----------
#     args : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     df : TYPE
#         DESCRIPTION.

#     '''
#     fname = os.path.join(data_loc,f"{args.dataset}_{tokenizer.__name__}.p")

#     args["Function used to read dataset"] = "load_tokenized_dataset"
#     args["Dataset pickle"] = fname
#     args["Tokenizer used"] = tokenizer.__name__
#     args["Tokenizer args"] = tokenizer_kwargs

#     if file_exists(fname):
#         print("Found pickle of tokenized dataset. Reading pickle.")
#         args["Dataset pickle found"] = "Yes"
#     else:
#         print(f"Pickle not found. Reading {args.dataset} dataset, tokenizing and saving pickle.")
#         args["Dataset pickle found"] = "No"
#         if args.dataset == "bbc":
#             location = get_data_path("datasets/bbc")
#         elif args.dataset == "bbcsport":
#             location = get_data_path("datasets/bbcsport")
#         df = pd.DataFrame(columns=["name","path","text","class"])
#         for classname in os.listdir(location):
#             class_dir_path = os.path.join(location, classname)
#             if os.path.isdir(class_dir_path):
#                 for filename in os.listdir(class_dir_path):
#                     file_path = os.path.join(class_dir_path, filename)
#                     with open(file_path, "r", encoding="utf8", errors="surrogateescape") as f:
#                         contents = str(f.read())
#                         df.loc[len(df)+1] = [filename, file_path, contents, classname]   

#         df["tokenized_text"] = df.text.apply(tokenizer, **tokenizer_kwargs)

#         df.to_pickle(fname)

#     return pd.read_pickle(fname), args



def cluster_members(cluster_num):
    global inverted_index
    list_of_occurrences = inverted_index[inverted_index.label == cluster_num]\
                        .sort_values("total_tf",ascending=False)["word"]
    words = []
    for word in list_of_occurrences:
        if word not in words:
            words.append(word)
    return words

from tabulate import tabulate

def print_clusters(cluster_nums, print_limit=20):
    if type(cluster_nums) == int:
        cluster_nums = [cluster_nums]
    tabled_results = []
    for cn in cluster_nums:
        print(cn)
        print("-"*len(str(cn)))
        print(", ".join(cluster_members(cn)[:print_limit]))
        print()

def to_pickle(what,where):
    pickle.dump(what, open(where,"wb"))
    
def read_pickle(where):
    return pickle.load(open(where,"rb"))


from sklearn.metrics.pairwise import paired_distances as sklearn_paired_distances
from sklearn.metrics.pairwise import euclidean_distances


def load_embeddings(vocab, word_vectors="word2vec_pretrained", path=None):
    
    '''
    Parameters
    ----------
    word_vectors : String. Default = "word2vec_pretrained"
        Method to load pre-trained word embeddings from a specified location. ["glove100", "glove300", "word2vec_pretrained"].
    file_path : Stirng
        Directory address of the pre-trained word embeddings to be loaded (except for the Word2Vec pre-trained model)
    vocab : List of strings
       .

    Returns
    -------
    embeddings : TYPE
        DESCRIPTION.

    '''

    if word_vectors == "glove100":

        # Load Embedding
        with open(os.path.expanduser(file_path)) as file:
            embeddings =  {
                                line.split()[0]:line.split()[1:] 
                                for line in file 
                                # add the selected words that are in the vocab
                                if line.split()[0] in vocab
                            }      

        embeddings = pd.DataFrame.from_dict(embeddings, orient="index").astype(float)

        embeddings = embeddings.apply(np.asarray, axis=1).rename("embedding")

    elif word_vectors == "glove300":

        # Load Embedding
        with open(os.path.expanduser(file_path)) as file:
            embeddings =  {
                                " ".join(line.split()[:-300]):line.split()[-300:] 
                                for line in file 
                                # add the selected words that are in the vocab
                                if line.split()[0] in vocab
                            }      

        embeddings = pd.DataFrame.from_dict(embeddings, orient="index").astype(float)

        embeddings = embeddings.apply(np.asarray, axis=1).rename("embedding")

    elif word_vectors == "word2vec_pretrained":
        import gensim.downloader as api

        # Load Pre-trained vectors
        wv = api.load('word2vec-google-news-300')   

        # Keep words that are present in pre-trained embeddings
        common_vocab = [word for word in vocab if word in wv]
        embeddings = wv[common_vocab]

        embeddings = pd.DataFrame(embeddings, index=common_vocab).astype(float)

        embeddings = embeddings.apply(np.asarray, axis=1).rename("embedding")  

    return embeddings.reindex(vocab).dropna()


def get_word2vec_model(data, word2vec_size, word2vec_window, word2vec_min_count, word2vec_epochs, word2vec_skip_gram):
    
    '''
    Parameters
    ----------
    data : 
    
    Returns
    -------
    embeddings : Word2Vec Model
        DESCRIPTION.

    '''

    tokenized_sentences = [sent for text in data.text.apply(Sentenced_Tokenizer) for sent in text]  
    
    print("Training Word2Vec")
    model = Gensim_Word2Vec(
                                sentences=tokenized_sentences, 
                                size=int(word2vec_size), 
                                window=int(word2vec_window), 
                                min_count=int(word2vec_min_count), 
                                workers=mp.cpu_count(),
                                iter=int(word2vec_epochs),
                                sg=int(word2vec_skip_gram)
                            )
    return model


def cluster(embeddings, wcde_clustering_method = , n_clusters = , linkage = , distance_threshold = ):
    
    '''
    Parameters
    ----------
    data : 
    
    Returns
    -------
    embeddings : Word2Vec Model
        DESCRIPTION.

    '''

    from sklearn.cluster import AgglomerativeClustering as AHC    

    print("Cluster labels and cluster centers not found. Clustering now.")
    if wcde_clustering_method == "kmeans":
        # KMeans
        clustering_model = KMeans(
                                        n_clusters=int(n_clusters),
                                        random_state=0,
                                        n_jobs=-1
                                   ).fit(list(embeddings.tolist()))

        labels = clustering_model.labels_
        cluster_centers = clustering_model.cluster_centers_

    elif wcde_clustering_method == "ahc" :
        # Agglomerative 
        if n_clusters and not pd.isnull(n_clusters) :
            print(f"Using AHC with specific number of clusters and {linkage} linkage")
            clustering_model = AHC(n_clusters=int(n_clusters), linkage=linkage).fit(embeddings.to_list())

        elif distance_threshold and not pd.isnull(distance_threshold) :
            print(f"Using AHC with Distance Threshold {distance_threshold} and {linkage} linkage")
            clustering_model = AHC(
                                        n_clusters=n_clusters,
                                        distance_threshold=float(distance_threshold),
                                        compute_full_tree=True,
                                        linkage=linkage
                                    ).fit(embeddings.to_list())            

            ahc_dt_clusters = clustering_model.n_clusters_

            if ahc_dt_clusters < 10:
                raise Exception("Not enough word clusters.")
        else:
            raise Exception(f"Ambiguous values : n_clusters={n_clusters}, distance_threshold={distance_threshold}. Exactly one of them has to be null.")

        labels = clustering_model.labels_ # One label for each word embedding

        # Collect clusters
        clusters = {label:[] for label in labels}
        for label, embedding in zip(labels,embeddings.to_list()):
            clusters[label].append(embedding)  

        cluster_centers = []
        for label in range(len(set(labels))):
            cluster_centers.append(np.mean(clusters[label], axis=0))

    return labels, cluster_centers, ahc_dt_clusters

def generate_embeddings(data, wec_df, cluster_function = , wcde_vector_normalize = True/False):
    
    '''
    Parameters
    ----------
    data : 
    
    Returns
    -------
    embeddings : Word2Vec Model
        DESCRIPTION.

    '''

    print("Generating Document Embeddings")
    N = len(data)
    if cluster_function == "qimin" or cluster_function == "fraj" or cluster_function == "tfidf_sum":
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

        if cluster_function == "qimin":
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
            
            if cluster_function == "qimin":            
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




if args.tokenizer == "regex":
    tokenizer = regex_tokenizer

"""### Generate Document Vectors"""
#PENDING - edit load_tokenized_dataset or write a new function
full_dataset, args = load_tokenized_dataset(args, tokenizer=tokenizer, removeStopwords=args.removeStopwords)

#PENDING - remove experiment names
if args.experiment_name =="wcde_w2v":
    #PENDING
    model = get_word2vec_model(full_dataset, [word2vec_size], [word2vec_window], [word2vec_min_count], [word2vec_epochs], [word2vec_skip_gram])

    # Get vocabulary of training set to filter the giant embedding object
    full_vocab = flatten(full_dataset["tokenized_text"], unique=True)

    vocab = [word for word in full_vocab if word in model.wv.vocab]

    # Get word embeddings
    embeddings = model.wv[vocab]

    # Make Pandas Series with index= words, values = embeddings 
    embeddings = pd.DataFrame(embeddings, index=vocab).astype(float)

    embeddings = embeddings.apply(np.asarray, axis=1).rename("embedding")

    embeddings.index.name = "word"

    print(f"{len(full_vocab)} unique words found in the data set")
    print(f"{len(vocab)} words from the data set taken as embeddings")
    
else:
    # Get vocabulary of training set to filter the giant embedding object
    vocab = flatten(full_dataset["tokenized_text"], unique=True)

    embeddings = load_embeddings(vocab = vocab, word_vectors= , path= )

    print(f"{len(vocab)} unique words found in the data set")

    print(f"{len(embeddings)} words from the data set found in the embeddings")


labels, cluster_centers, args_ahc_dt_clusters = cluster(embeddings = embeddings, wcde_clustering_method = , n_clusters = , linkage = , distance_threshold = )

len(cluster_centers)

# Cluster Distribution
pd.Series(labels).value_counts().sort_values(ascending=False).reset_index(drop=True).plot(figsize=(15,5))

wec_df = embeddings.to_frame()

# Name of index
wec_df.index.name = "word"

# Add Labels
wec_df["label"] = labels

# Get cluster center for each word
cluster_centers_df = pd.DataFrame(cluster_centers).apply(np.asarray, axis=1).to_frame(name="cluster_center")

wec_df = wec_df.join(cluster_centers_df, on="label")  

list_of_text_embeddings = generate_embeddings(data = full_dataset, wec_df = wec_df, cluster_function = , wcde_vector_normalize = True/False)