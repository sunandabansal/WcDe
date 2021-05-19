# -*- coding: utf-8 -*-

'''

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''

# Importing Libraries
import os      
import pdb 
import argparse
import numpy as np
import pandas as pd

import helpers
import WcDe

# For clustering documents - Task specific imports
import sklearn


def read_bbc_dataset(path):
    '''
    It reads any of the BBC datasets (BBC or BBCSport). The raw text files can be 
    downloaded from http://mlg.ucd.ie/datasets/bbc.html. The zipped files can be 
    unzipped to get the folders that contain the entire bbc or bbc sports dataset.
    These folders are then parsed to get the texts and corresponding classes.
    
    Parameters
    ----------
    location : str
        Path to the bbc or bbcsport folder containing raw documents.

    Returns
    -------
    texts : list
        List of textual content of all documents in the dataset.
    classes : list
        List of class labels corresponding to each textual document.

    '''
    
    texts, classes = [], []
    
    for class_label in os.listdir(path):
        class_dir_path = os.path.join(path, class_label)
        
        # If its a directory, then it is considered to be the class label
        # and all the files in the directory are read and associated with 
        # this class label
        if os.path.isdir(class_dir_path):            
            for file_name in os.listdir(class_dir_path):
                file_path = os.path.join(class_dir_path, file_name)
                with open(file_path, "r", encoding="utf8", errors="surrogateescape") as f:
                    texts.append(str(f.read()))
                    classes.append(class_label)  
                    
    return texts, classes

def read_glove_embeddings(path, vocab, vector_size):
    '''
    Reads GloVe pre-trained word embeddings into a Pandas Series

    Parameters
    ----------
    path : str
        Path to GloVe pre-trained embedding file.
    vocab : list or None
        List of terms to get word vectors for.
    vector_size : int
        Size of a word vector in the word embedding.

    Returns
    -------
    embeddings : pandas.Series
        Pandas Series with words as the index and vector (numpy array) as the corresponding value.

    '''
    embeddings = {}
    
    with open(os.path.expanduser(path)) as file:
        for line in file:
            word    = " ".join(line.split()[:-vector_size]) 
            vector  = line.split()[-vector_size:]
            
            if vocab is None or word in vocab:
                embeddings[word] = vector    

    embeddings = pd.DataFrame.from_dict(embeddings, orient="index").astype(float)

    embeddings = embeddings.apply(np.asarray, axis=1).rename("vector")
    
    embeddings = embeddings.reindex(vocab).dropna()
    
    embeddings.index.name = "word"
    
    return embeddings
        
def parse_command_line_arguments(desc=""):

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
                            '--dataset_path', 
                            type=str, 
                            help='Path to BBC datasets - path/to/bbc[sport]/'
                       )

    parser.add_argument(
                            '--embedding_file', 
                            type=str,
                            help='Path to GloVe 100d file - /path/to/glove.6B.100d.txt'
                       )

    parser.add_argument(
                            '--clustering_algorithm', 
                            type=str,
                            default="ahc",
                            help='''Only "kmeans" and "ahc" supported.'''
                       )

    parser.add_argument(
                            '--distance_threshold', 
                            type=float,
                            default=8.0,
                            help='''AHC Distance Threshold'''
                       )

    parser.add_argument(
                            '--linkage', 
                            type=str,
                            default="ward",
                            help='''"average", "complete" or "ward"'''
                       )

    parser.add_argument(
                            '--n_clusters', 
                            type=int,
                            default=None,
                            help='''AHC or K-Means number of clusters'''
                       )

    return parser.parse_args()

if __name__ == "__main__":

    # Read demo variables
    args = parse_command_line_arguments()
    
    # Read Dataset
    print("Reading dataset.")
    texts, classes = read_bbc_dataset(path=args.dataset_path)       
    
    # Tokenized the texts
    print("Tokenizing documents.")
    tokenized_texts = [helpers.tokenize(text) for text in texts]
    
    # Get the vocabulary of dataset
    vocab = helpers.flatten(tokenized_texts, unique=True)
    
    # Get word vectors (pandas.Series)
    print("Getting word vectors.")
    word_vectors = read_glove_embeddings(path=args.embedding_file, vocab=vocab, vector_size=100)
    
    # Cluster Word Vectors
    print("Clustering word vectors.")
    cluster_labels = WcDe.cluster_word_vectors(
                                                    word_vectors=word_vectors,
                                                    clustering_algorithm=args.clustering_algorithm,
                                                    n_clusters=args.n_clusters,
                                                    distance_threshold=args.distance_threshold,
                                                    linkage=args.linkage
                                              )

    # Generate Document Vectors
    print("Generating document vectors.")
    wcde_doc_vectors = WcDe.get_document_vectors(    
                                                    tokenized_texts,
                                                    word_vectors=word_vectors,
                                                    cluster_labels=cluster_labels,
                                                    weight_function="cfidf",
                                                    normalize=True
                                                )
    # Task - Cluster Documents
    print("Clustering document vectors.")
    document_clustering_model = sklearn.cluster.KMeans(n_clusters=5, random_state=0)
    document_clustering_model = document_clustering_model.fit(wcde_doc_vectors)
    document_cluster_label = list(document_clustering_model.labels_)
    
    # Evaluation
    score = sklearn.metrics.normalized_mutual_info_score(classes, document_cluster_label)    
    print("Performance (NMI):", score)