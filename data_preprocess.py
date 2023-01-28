import pandas as pd
import numpy as np
import json
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

nltk.download("stopwords")
nltk.download('punkt')

def tokenize(data):
    # Word tokenize
    tokens = word_tokenize(data)
    return tokens

def remove_junks(tokens):
    # Removing punctuations
    wo_punctuations  = set(tokens).difference(set(punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    wo_stopwords_puncts = wo_punctuations.difference(stop_words)
    return list(wo_stopwords_puncts)

def stemming(clean_words):
    stem_obj = PorterStemmer()
    lower_case_words = list(map(lambda x:x.lower(),clean_words))
    stem_words = list(map(lambda x:stem_obj.stem(x),lower_case_words))
    return stem_words

def create_vocabulary(data, column):
    # creating a vocabulary out of text column
    vocab = []
    for record in data[column].values:
      vocab.extend(record)
    return set(vocab)

def clean_data(df, column):
    df[column]= df[column].apply(tokenize)
    df[column]= df[column].apply(remove_junks)
    df[column]= df[column].apply(stemming)
    return df

# Let's represent each text with bag-of-words method
def bag_of_words_representation(df, column):
    with open("model/vocabulary.json","r") as voc_save_file:
        vocab_dict = json.load(voc_save_file)
    vocab = vocab_dict["vocabulary"]
    def vector_represent(record):
      bag_of_words = np.zeros(len(vocab))
      for word in record:
        if word in vocab:
          bag_of_words[vocab.index(word)] += 1
      return bag_of_words
      
    vector_matrix = list(map(lambda x: vector_represent(x),df[column].values))
    bow_df = pd.DataFrame(vector_matrix,columns=vocab)
    df.drop(columns=[column], inplace=True)
    final_df = pd.concat([df,bow_df],axis=1)
    return final_df