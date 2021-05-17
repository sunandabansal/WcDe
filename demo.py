# -*- coding: utf-8 -*-

'''

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''

# Importing Libraries
import os
import nltk
import numpy as np
import pandas as pd

import helpers
import WcDe

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
            
            if vocab is not None and word in vocab:
                embeddings[word] = vector    

    embeddings = pd.DataFrame.from_dict(embeddings, orient="index").astype(float)

    embeddings = embeddings.apply(np.asarray, axis=1).rename("embedding")
    
    embeddings = embeddings.reindex(vocab).dropna()
    
    embeddings.index.name = "word"
    
    return embeddings
        

if __name__ == "__main__":
    
    # Read Dataset
    texts, classes = read_bbc_dataset(path="/path/to/bbc/") 
    
    # Tokenized the texts
    tokenized_texts = [helpers.tokenize(text) for text in texts]
    
    # Get the vocabulary of dataset
    vocab = helpers.flatten(tokenized_texts, unique=True)
    
    # Get word vectors 
    word_vectors = read_glove_embeddings(path="/path/to/glove.6B.100d.txt", vocab=vocab, vector_size=100)
    

    cluster_labels = WcDe.cluster_word_vectors(
                                                  word_vectors=word_vectors, 
                                                  clustering_algorithm="kmeans",
                                                  n_clusters=1500 
                                              )
    
    wcde_doc_vectors = WcDe.get_document_vectors(    
                                                    tokenized_texts,
                                                    word_vectors=word_vectors,
                                                    cluster_labels=cluster_labels,
                                                    weight_function="cfidf",
                                                    normalize=True
                                                )