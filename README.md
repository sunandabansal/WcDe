# Word Clusters based Document Embedding (WcDe) - A demo


[![CC BY 4.0][license-shield]][license-url]

This repository provides implementation for generating Word Clusters based Document Embedding (WcDe). The purpose of this repository, at the moment, is to allow the user to experiment with the methodology. Therefore, at the moment, the demo is only configured to work for BBC datasets with 100-dimensional pre-trained GloVe embedding. This demo generates WcDe document representations, clusters them and evaluates the performance based on the Normalized Mutual Information score.

## Files and Methods
1. `demo.py` - This file is to be run for the demo. It contains the methods and parameters that are specific to this demo. 
    1. `__main__` - The main body of `demo.py` that sets the demo parameters, generates document vectors, clusters document vectors and evaluates the performance using Normalized Mutual Information between the clusters and the true class of documents. The parameters that can be set for the demo are - 
 
        | Variable             | Default Value                | Type        | Comment                                                                                                      |
        |----------------------|------------------------------|-------------|--------------------------------------------------------------------------------------------------------------|
        | dataset_path         | "/path/to/bbc"               | str         | Path to `bbc` or `bbcsport` directories                                                                      |
        | embedding_file       | "/path/to/glove.6B.100d.txt" | str         | Word embedding to be used                                                                                    |
        | word_vector_size     | 100                          | int         | Size of each word vectors                                                                                    |
        | clustering_algorithm | "ahc"                        | str         | The clustering technique. Acceptable values are - "ahc", "kmeans".                                           |
        | linkage              | "ward"                       | str         | Merge Criteria for Hierarchical Clustering. Acceptable values are - "ward", "complete", "average", "single". |
        | n_clusters           | None                         | int or None | Number of clusters for Flat clustering                                                                       |
        | distance_threshold   | 8                            | float       | Distance Threshold for Hierarchical Clustering                                                               |
        | weighting_scheme     | "cfidf"                      | str         | The weighting scheme to be used to calculate score of word cluster in the document. Acceptable values are    |
        | length_normalize     | True                         | bool        | Whether to length normalize the WcDe document vector or not                                                  |
    2. `read_bbc_dataset()` - It reads any of the BBC datasets (BBC or BBCSport). The __raw__ text files can be downloaded from [http://mlg.ucd.ie/datasets/bbc.html](http://mlg.ucd.ie/datasets/bbc.html). The zipped file can be unzipped to get the folders the raw data in the form of two directories - `bbc` or `bbcsport` dataset. This method takes the path of one directory and parses its contents to get the texts and corresponding classes. To make the experiment deterministic, the documents are read in the alphabetical order of class name. For the documents of same class the documents are sorted in the alphabetical order of file names.
    3. `read_glove_embeddings()` - Reads GloVe pre-trained word embeddings and returns a list of words and an array containing word vectors corresponding to the words. To make the experiment deterministic, the words are sorted in alphabetical order.
1. `WcDe.py` - This file contains the methods that implements the WcDe methodology.
    1. `cluster_word_vectors()` - Clusters the word vectors.
    2. `get_document_vectors()` - Generates the WcDe document vectors.
1. `helpers.py` - This file contains additional helper methods.
    1. `tokenize()` - Tokenizes a piece of text.
    2. `flatten()` - Flattens nested lists.



## Getting Started

To run the demo follow these simple steps.

### Prerequisites

* Python >= 3.6
* `virtualenv` (highly recommended)  

*For information on how to install `vistualenv`, please refer to - [Python - Installing packages using pip and virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)*

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/sunandabansal/WcDe
   cd WcDe
   ```
2. __Set up Virtual Environment__ (highly recommended)  
   Create virtual environment
    ```sh
    virtualenv env
    ```
   Activate virtual environment
    ```sh
    source env/bin/activate
    ```
3. Install packages
    ```sh
    pip3 install -r requirements.txt
    ```
   

### Usage Instructions

1. In `demo.py`, set the the following values in the main body. 

    | Variable          | Comment                                                   |
    |-------------------|-----------------------------------------------------------|
    | dataset_path      | Path to `bbc` or `bbcsport` directories                   |
    | embedding_file    | Path to GloVe 100-dimensional pre-trained word embedding  |
    | word_vector_size  | Size of each word vectors                                 |
    
    ```py
    dataset_path      = "your/path/to/bbc"
    embedding_file    = "your/path/to/glove.6B.100d.txt"
    word_vector_size  = 100
    ```
2. Run
    ```sh
    python3 demo.py
    ```
    
### Expected Output

Using Glove 100d word embedding and the default clustering configurations given below -

| Variable             | Value  | Comment                                        |
|----------------------|--------|------------------------------------------------|
| clustering_algorithm | "ahc"  | The clustering technique                       |
| linkage              | "ward" | Merge Criteria for Hierarchical Clustering     |
| n_clusters           | None   | Number of clusters for Flat clustering         |
| distance_threshold   | 8      | Distance Threshold for Hierarchical Clustering |

The output for __BBC Dataset__-

```
Reading dataset.
Tokenizing documents.
Getting word vectors.
Clustering word vectors.
Generating document vectors.
Clustering document vectors.
Performance (NMI): 0.7956132539056411
```
The output for __BBC Sport Dataset__-

```
Reading dataset.
Tokenizing documents.
Getting word vectors.
Clustering word vectors.
Generating document vectors.
Clustering document vectors.
Performance (NMI): 0.8004845766270388
```


## License

[![CC BY 4.0][license-shield]][license-url]

Distributed under the [Creative Commons Attribution 4.0 International License][cc-by] License.  
See `LICENSE` for more information.

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[license-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-informational.svg?style=for-the-badge
[license-url]: https://github.com/sunandabansal/WcDe/LICENSE
