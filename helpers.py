#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

"""
import nltk
import os

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

def tokenize(
    text, 
    tokenizer=nltk.tokenize.RegexpTokenizer(r'(?u)\b\w\w+\b').tokenize,
    lowercase=True,
    exclude=None,
    **tokenizer_kwargs
    ):
    '''
    Return a list of tokens extracted from a given text 

    Parameters
    ----------
    text : str
        Text document to tokenize.
    tokenizer : function, optional
        Function to be used for tokenization. The method should accept the text document 
        as the first positional argument. The rest of the arguments can be passed as keyword
        arguments to this function.
        The default is nltk.tokenize.RegexpTokenizer(r'(?u)\b\w\w+\b').tokenize. Default tokenizer
        extracts all the alphanumeric sequences containing more than 1 character.
    lowercase : bool, optional
        If True, the text will be casefolded before tokenization. The default is True.
    exclude : list or None, optional
        The list of terms to exclude from tokens. This could be stopwords or more. 
        If None, or empty list is provided, no tokens will be excluded from the results.
        The default is None.
    tokenizer_kwargs : dict
        Any additional keyword arguments passed to this function are passed to the tokenizer, if any.

    Returns
    -------
    tokens : list
        List of tokens, in order, including multiple occurrences.

    '''
    if lowercase: text = text.lower()

    tokens = tokenizer(text,**tokenizer_kwargs)

    if exclude is not None and len(exclude > 0):
        tokens = [token for token in tokens if not token in exclude]

    return tokens