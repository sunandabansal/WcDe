# -*- coding: utf-8 -*-

'''

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''

# Importing Libraries
import os  
import pandas as pd  

# Importing Demo Modules
import helpers
import WcDe
import demo

# For clustering documents - Task specific imports
import sklearn

    import pandas as pd  

    args = pd.Series()
    args["dataset_path"] = "/media/Sunanda/Mask/Work/datasets/bbc"
    args["embedding_file"] = "/media/Sunanda/Mask/Work/common/embeddings/glove.6B.100d.txt"
    
    args["clustering_algorithm"] = "ahc"
    args["linkage"] = "ward"
    args["n_clusters"] = None
    args["distance_threshold"] = 8

    print("Reading tokenized pickled dataset - intermediate_files/full_clustering/bbc_regex_tokenizer.p")
    dataset = pd.read_pickle("/media/Sunanda/Mask/Work/WeWcDe/intermediate_files/full_clustering/bbc_regex_tokenizer.p")
    print(dataset.head())
    print(dataset.columns)

    print("Reading pickled wec_df - ahc_8_ward_g100_ncfidf/wec_df_bbc.p")
    wec_df = pd.read_pickle("/home/sunanda/Documents/sunanda/WeWcDe Experiments/full_clustering/data/heatmaps/data/ahc_8_ward_g100_ncfidf/wec_df_bbc.p")
    print(wec_df.head())
    print(wec_df.columns)

    # Read Dataset
    print("\n")
    print("Reading dataset.")
    texts, classes = read_bbc_dataset(path=args.dataset_path)    

    # Tokenize
    print("Tokenizing Using Implemented Tokenizer.")
    tokenized_texts = [helpers.tokenize(text) for text in texts] 

    # Get the vocabulary of dataset
    print("Getting Vocab.")
    vocab = helpers.flatten(tokenized_texts, unique=True)
    vocab_2 = helpers.flatten(dataset.tokenized_text, unique=True)

    diff2 = set(vocab) - set(vocab).intersection(set(vocab_2))
    print(len(vocab), len(vocab_2), len(diff2))
    print(diff2)

    if len(vocab) != len(vocab_2):
        print("Vocabulary is different.")
        return 

    # Get word vectors (pandas.Series)
    print("\n")
    print("Getting word vectors.")
    words, word_vectors = read_glove_embeddings(path=args.embedding_file, vocab=vocab, vector_size=100)
    print("Embeddings shape (Demo)",np.array(word_vectors).shape)

    print("Comparing the words found.")
    words_2 = list(wec_df.index)
    diff2 = set(words) - set(words).intersection(set(words_2))
    print(len(vocab), len(words), len(words_2), len(diff2))
    print(diff2)

    print("Comparing embeddings.")
    # Use .loc instead of .reindex because .loc will throw error if the word doesn't exist in index 
    try:
        embedding_diff = np.array(wec_df.loc[words]["embedding"].apply(np.asarray).tolist()).astype(float) - np.array(word_vectors).astype(float)    
    except:
        pdb.set_trace()

    print("Difference is zero?", np.all(embedding_diff==0))

    if not np.all(embedding_diff==0):
        print("The embeddings are different.")
        return 

    # Cluster Word Vectors
    print("\n")
    print("Clustering word vectors.")
    cluster_labels = WcDe.cluster_word_vectors(
                                                    word_vectors=word_vectors,
                                                    clustering_algorithm=args.clustering_algorithm,
                                                    n_clusters=args.n_clusters,
                                                    distance_threshold=args.distance_threshold,
                                                    linkage=args.linkage
                                              )

    print("Number of Clusters", len(set(cluster_labels)))

    labels_nmi = sklearn.metrics.normalized_mutual_info_score(wec_df["label"].loc[words], cluster_labels)    
    print("Clustering label consensus (NMI):", labels_nmi)

    if labels_nmi != 1:
        print("The word clustering are different.")
        return

    # Generate Document Vectors
    print("\n")
    print("Generating document vectors. - DEMO VERSION")
    wcde_doc_vectors = WcDe.get_document_vectors_v2(    
                                                        tokenized_texts,
                                                        words=words,
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

    # Generate Document Vectors
    print("\n")
    print("Generating document vectors. - DEMO + Pickled Dataset")
    wcde_doc_vectors = WcDe.get_document_vectors_v2(    
                                                        dataset["tokenized_text"],
                                                        words=words,
                                                        cluster_labels=cluster_labels,
                                                        weight_function="cfidf",
                                                        normalize=True,
                                                        prefix="demopickdf"
                                                    )

    # Task - Cluster Documents
    print("Clustering document vectors.")
    document_clustering_model = sklearn.cluster.KMeans(n_clusters=5, random_state=0)
    document_clustering_model = document_clustering_model.fit(wcde_doc_vectors)
    document_cluster_label = list(document_clustering_model.labels_)
    
    # Evaluation
    score = sklearn.metrics.normalized_mutual_info_score(dataset["class"], document_cluster_label)    
    print("Performance (NMI):", score)
def test_components():
    
    # Read demo variables
    # dataset_path    = "/path/to/bbc"
    embedding_file  = "/media/Sunanda/Mask/Work/common/embeddings/glove.6B.100d.txt"
    
    clustering_algorithm    = "ahc"
    linkage                 = "ward"
    n_clusters              = None
    distance_threshold      = 8
    
    # Read Pickled Dataset        
    print("Reading tokenized pickled dataset - intermediate_files/full_clustering/bbc_regex_tokenizer.p")
    dataset = pd.read_pickle("/media/Sunanda/Mask/Work/WeWcDe/intermediate_files/full_clustering/bbc_regex_tokenizer.p")

    texts, classes = dataset["text"], dataset["class"]     
    
    # Tokenized the texts
    print("Tokenizing documents.")
    tokenized_texts = [helpers.tokenize(text) for text in texts]
    
    # Get the vocabulary of dataset
    vocab = helpers.flatten(tokenized_texts, unique=True)
    
    # Get word vectors (pandas.Series)
    print("Getting word vectors.")
    words, word_vectors = demo.read_glove_embeddings(path=embedding_file, vocab=vocab, vector_size=100)
    
    # Cluster Word Vectors
    print("Clustering word vectors.")
    cluster_labels = WcDe.cluster_word_vectors(
                                                    word_vectors=word_vectors,
                                                    clustering_algorithm=clustering_algorithm,
                                                    n_clusters=n_clusters,
                                                    distance_threshold=distance_threshold,
                                                    linkage=linkage
                                              )

    # Generate Document Vectors
    print("Generating document vectors.")
    wcde_doc_vectors = WcDe.get_document_vectors(    
                                                    tokenized_texts,
                                                    words=words,
                                                    cluster_labels=cluster_labels,
                                                    weight_function="cfidf",
                                                    normalize=True
                                                )
    
    # Task - Cluster Documents
    print("Clustering document vectors.")
    document_clustering_model = sklearn.cluster.KMeans(
                                                            n_clusters=5, 
                                                            random_state=0
                                                       )
    document_clustering_model = document_clustering_model.fit(wcde_doc_vectors)
    document_cluster_label = list(document_clustering_model.labels_)
    
    # Evaluation
    score = sklearn.metrics.normalized_mutual_info_score(classes, document_cluster_label)    
    print("Performance (NMI):", score)

def demo():

    # Read demo variables
    dataset_path    = "/path/to/bbc"
    embedding_file  = "/path/to/glove.6B.100d.txt"
    
    clustering_algorithm    = "ahc"
    linkage                 = "ward"
    n_clusters              = None
    distance_threshold      = 8
    
    # Read Dataset
    print("Reading dataset.")
    texts, classes = read_bbc_dataset(path=dataset_path)       
    
    # Tokenized the texts
    print("Tokenizing documents.")
    tokenized_texts = [helpers.tokenize(text) for text in texts]
    
    # Get the vocabulary of dataset
    vocab = helpers.flatten(tokenized_texts, unique=True)
    
    # Get word vectors (pandas.Series)
    print("Getting word vectors.")
    words, word_vectors = read_glove_embeddings(path=embedding_file, vocab=vocab, vector_size=100)
    
    # Cluster Word Vectors
    print("Clustering word vectors.")
    cluster_labels = WcDe.cluster_word_vectors(
                                                    word_vectors=word_vectors,
                                                    clustering_algorithm=clustering_algorithm,
                                                    n_clusters=n_clusters,
                                                    distance_threshold=distance_threshold,
                                                    linkage=linkage
                                              )

    # Generate Document Vectors
    print("Generating document vectors.")
    wcde_doc_vectors = WcDe.get_document_vectors(    
                                                    tokenized_texts,
                                                    words=words,
                                                    cluster_labels=cluster_labels,
                                                    weight_function="cfidf",
                                                    normalize=True
                                                )
    
    # Task - Cluster Documents
    print("Clustering document vectors.")
    document_clustering_model = sklearn.cluster.KMeans(
                                                            n_clusters=5, 
                                                            random_state=3
                                                       )
    document_clustering_model = document_clustering_model.fit(wcde_doc_vectors)
    document_cluster_label = list(document_clustering_model.labels_)
    
    # Evaluation
    score = sklearn.metrics.normalized_mutual_info_score(classes, document_cluster_label)    
    print("Performance (NMI):", score)

# if __name__ == "__main__":
    
