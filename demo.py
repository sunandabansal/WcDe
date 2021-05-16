# -*- coding: utf-8 -*-

'''

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

'''
# Importing Libraries
import numpy as np
import pandas as pd

import WcDe

if args.tokenizer == "regex":
    tokenizer = regex_tokenizer

"""### Generate Document Vectors"""
#PENDING - edit load_tokenized_dataset or write a new function
full_dataset, args = WcDe.load_tokenized_dataset(args, tokenizer=tokenizer, removeStopwords=args.removeStopwords)

#PENDING - remove experiment names
if args.experiment_name =="wcde_w2v":
    #PENDING
    model = WcDe.get_word2vec_model(full_dataset, [word2vec_size], [word2vec_window], [word2vec_min_count], [word2vec_epochs], [word2vec_skip_gram])

    # Get vocabulary of training set to filter the giant embedding object
    full_vocab = WcDe.flatten(full_dataset["tokenized_text"], unique=True)

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
    vocab = WcDe.flatten(full_dataset["tokenized_text"], unique=True)

    embeddings = WcDe.load_embeddings(vocab = vocab, word_vectors= , path= )

    print(f"{len(vocab)} unique words found in the data set")

    print(f"{len(embeddings)} words from the data set found in the embeddings")


labels, cluster_centers, args_ahc_dt_clusters = WcDe.cluster(embeddings = embeddings, wcde_clustering_method = , n_clusters = , linkage = , distance_threshold = )

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

list_of_text_embeddings = WcDe.generate_embeddings(data = full_dataset, wec_df = wec_df, cluster_function = , wcde_vector_normalize = True/False)