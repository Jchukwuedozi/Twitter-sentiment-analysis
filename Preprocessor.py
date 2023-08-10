from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import numpy as np
import pandas as pd

abbreviations = {'dem': 'them', 'dey': 'they', 'd': 'the', 'btw': 'by the way',
'pov': 'point of view', 'nd': 'and', '&amp;': 'and', '&': 'and', 'info': 'information', 
'nyt': 'night', 'lyt': 'light', 'stee': 'still', 'tyt': 'tight', 'dem': 'them', 
'2moro': 'tomorrow','b': 'be', 'myt': 'might', 'pple': 'people', 'gud': 'good', 
'ryt': 'right', 'tym': 'time', 'luk': 'look', 'b4': 'before', 'lyk': 'like', 'u': 'you'}


def remove_abbreviations(word):
    ''' This function removes/replaces abbreviated words, 
        so as to reduce the number of features in the document.
        
        Args: 
                word - This is the document it's expected to work on, and it must be a list of strings.
    '''
    if word in abbreviations.keys():
        new_word = abbreviations[word]
        return new_word
    else:
        return word
    
def remove_url(word):
    pattern = r'https*://\w+\.\w+(\.\w+(/\w+)+|/\w+|\.\w+)'
    return re.sub(pattern, '', word)

def prepare_data(data):
    stop_words = stopwords.words('english')
    new_data = list(data)
    sentences = [' '.join([str(TextBlob(remove_url(remove_abbreviations(word))).correct()) for word in sentence.split() if word not in stop_words]) for sentence in new_data]
    return sentences 
        

        
    
        
              
class Preprocessed(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self


    def transform(self, X):
        stop_words = stopwords.words('english')
        X = X['Text']
        X = list(X)
        X = [' '.join([remove_url(remove_abbreviations(word)) for word in sentence.split()]) for sentence in X]
#         X = [' '.join([str(TextBlob(remove_url(remove_abbreviations(word))).correct()) for word in sentence.split()]) for sentence in X]
        X = pd.DataFrame(X, columns=['Text'])
        return X    
            


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.feature_names]
        print(X.loc[:,self.feature_names])
        return X
        
             
                
                

            







