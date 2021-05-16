#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author  :   Sunanda Bansal (sunanda0343@gmail.com)
Year    :   2021

"""
import nltk

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
