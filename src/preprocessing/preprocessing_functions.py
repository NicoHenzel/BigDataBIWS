# Description: This script contains functions for text preprocessing.

# Import libraries
import pandas as pd
import string
import nltk
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')



# Function to remove punctuation
def keep_only_words(text):
    """
    This function removes punctuation from a text.
    
    Parameters
    ----------
    text : str
        Text to be cleaned.

    Returns
    -------
    only_words : str
        Text without punctuation.
    """
    only_words=text.translate(str.maketrans('', '', string.punctuation))
    return only_words



# Function for tokenization
def tokenize_to_list(text):
    """
    This function tokenizes a text into a list of words.

    Parameters
    ----------
    text : str
        Text to be tokenized.

    Returns
    -------
    tokens : list
        List of words.
    """
    tokens=word_tokenize(text)
    return tokens



# Function to determine text language
def language_detection(text):
    """
    This function detects the language of a text.

    Parameters
    ----------
    text : list
        Text to be analyzed.

    Returns
    -------
    english : bool
        True if text is english, False if not.
    """
    if detect(text) == 'en':
        return True
    else:
        return False



# Function for language filtering
def filter_language(text):
    """
    This function filters a text for english words.

    Parameters
    ----------
    text : list
        Text to be filtered.

    Returns
    -------
    english_tokens : list
        List of english words.
    """
    english_words = set(nltk.corpus.words.words())
    english_tokens=[]
    text_lower = [word.lower() for word in text]
    for word in text_lower:
        if word in english_words:
            english_tokens.append(word)
    return english_tokens



# Function for POS tagging
def token_and_tag_to_list(tokens):
    """
    This function tags a list of words with their POS.

    Parameters
    ----------
    tokens : list
        List of words to be tagged.

    Returns
    -------
    token_and_tag : list
        List of words and their POS.
    """

    # Use nltk.pos_tag to tag the tokens with POS tags
    tokens_pos = nltk.pos_tag(tokens)
    
    # Initialize an empty list to store tokens and their tags
    token_and_tag = []

    # Iterate over the list of upper_tokens
    for word, tag in tokens_pos:
        # For each token, append a tuple with the token and its POS tag to token_and_tag
        token_and_tag.append((word.upper(), tag))

    # Return the list of tuples
    return token_and_tag



# Function to lemmatize
def lemmatize_to_list(tokens):
    """
    This function lemmatizes a list of words.

    Parameters
    ----------
    tokens : list
        List of words to be lemmatized.

    Returns
    -------
    lemma_list : list
        List of lemmatized words.
    """
    lemmatizer = WordNetLemmatizer()
    lemma_list=[]
    for token in tokens:
        lemma_list.append(lemmatizer.lemmatize(token))
    return lemma_list



# Function to remove stop words
def non_stopword_token_to_list(tokens):
    """
    This function removes stop words from a list of words.

    Parameters
    ----------
    tokens : list
        List of words to be cleaned.

    Returns
    -------
    non_stopword_tokens : list
        List of words without stop words.
    """
    stop_words = set(stopwords.words('english'))
    non_stopword_tokens=[]
    for token in tokens:
        if token not in stop_words:
            non_stopword_tokens.append(token)
    return non_stopword_tokens



# Function to stem
def stemmed_token_to_list(tokens):
    """
    This function stems a list of words.

    Parameters
    ----------
    tokens : list
        List of words to be stemmed.

    Returns
    -------
    stemmed_tokens : list
        List of stemmed words.
    """
    #Create a list of upper tokens
    upper_tokens=[token.upper() for token in tokens]
    #Create a list of stemmed tokens
    stemmed_tokens=[]
    for token in upper_tokens:
        stemmed_tokens.append(PorterStemmer().stem(token))
    return stemmed_tokens



# Function to convert to uppercase
def upper_tokens_to_list(tokens):
    """
    This function converts a list of words to uppercase.

    Parameters
    ----------
    tokens : list
        List of words to be converted.

    Returns
    -------
    upper_tokens : list
        List of words in uppercase.
    """
    upper_tokens=[token.upper() for token in tokens]
    return upper_tokens

# Error when importing this function
# # Function to remove all 'NONE' entries in a dataframe column
# def remove_none_entries(column):
#     """
#     This function removes all 'NONE' entries from a dataframe column containing lists as entries.

#     Parameters
#     ----------
#     tokens : Dataframe column with list entries
#         Dataframe column to be cleaned.

#     Returns
#     -------
#     tokens : list
#         List without 'NONE' entries.
#         Technically this function returns None, the change is inplace.
#     """
#     for lst in column:
#         # check if 'NONE' is in lst
#         if 'NONE' in lst:
#             # remove all 'NONE' entries from lst
#             while 'NONE' in lst:
#                 lst.remove('NONE')
#     return None


# Function to run the entire pipeline (shouldn't be used imo)
def text_preprocessing_pipeline(text):
    """
    This function runs the entire text preprocessing pipeline.

    Parameters
    ----------
    text : str
        Text to be cleaned.

    Returns
    -------
    df_prep : DataFrame
        DataFrame with columns for each step of the pipeline.
    """
    only_words=keep_only_words(text)
    df_prep=pd.DataFrame()
    df_prep['Token']=tokenize_to_list(only_words)
    df_prep['Filtered Language']=filter_language(df_prep['Token'])
    df_prep['Token and POS-Tag']=token_and_tag_to_list(df_prep['Filtered Language'])
    df_prep['Lemma']=lemmatize_to_list(df_prep['Filtered Language'])
    df_prep['Non-Stopword Token']=non_stopword_token_to_list(df_prep['Lemma'])
    df_prep['Stemmed Token']=stemmed_token_to_list(df_prep['Non-Stopword Token'])
    df_prep['Upper Token']=upper_tokens_to_list(df_prep['Stemmed Token'])
    return df_prep
