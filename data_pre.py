# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:54:11 2021

@author: jackw
"""

#%% Set Up

# imports
import nltk
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')

# instantiate stopword and lemmatizer objects
stopwords = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()

# import data. Columns = (Author, Article Title, Source Title, Abstract, Publication Year, DOI, Projected or Documented)
data = pd.read_csv(r"C:\Users\jackw\Desktop\FiCli\Data_final.csv", encoding='unicode_escape')

#%% text clean function
'''
takes text and returns lemmatized text with no punctuation in a tokenized list format
@param text, raw text to be formatted
@return text, formatted text
'''
def clean_text_tfidf(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text

'''
takes text and returns lemmatized text with no punctuation in a non-tokenized format
@param text, raw text to be formatted
@return text, formatted text
'''
def clean_text_ngrams(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords])
    #text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text

#%%
'''
function to take the projected, documented, and projected and documented identifiers and change them to a binary format
@param ident, supplied identifier of the article
@return bool, whether the article is included in FiCli/projected to be or is not
'''
def set_class(ident):
    if ((ident == 'PROJECTED') | (ident == 'DOCUMENTED') | (ident == 'PROJECTED AND DOCUMENTED')):
        return True
    else:
        return False
    
#%%
# Get column of full text
data['Text'] = data.apply(lambda row: str(row['Article Title']) + " " + str(row['Abstract']), axis=1)

# Set identifiers to binary status
data['Included'] = data['Projected or Documented'].apply(lambda x: set_class(x))

#%%
# Apply cleaning function 
data['Clean Text TFIDF'] = data['Text'].apply(lambda x: clean_text_tfidf(x))
data['Clean Text n-gram'] = data['Text'].apply(lambda x: clean_text_ngrams(x))


#%%
# Create vectorized data (uni-gram tf idf, n=1)
tfidf_vect = TfidfVectorizer(analyzer=clean_text_tfidf)
X_tfidf_uni = tfidf_vect.fit_transform(data['Text'])
X_features_uni = pd.DataFrame(X_tfidf_uni.toarray())
X_features_uni.columns = tfidf_vect.get_feature_names()

#%%
# Create vectorized data (uni-gram tf idf, n=2)
vect_bi = TfidfVectorizer(ngram_range=(2,2))
X_bi = vect_bi.fit_transform(data['Clean Text n-gram'])
X_features_bi = pd.DataFrame(X_bi.toarray())
X_features_bi.columns = vect_bi.get_feature_names()

#%%
# Create vectorized data (uni-gram tf idf, n=3)
vect_both = TfidfVectorizer(ngram_range=(1,2))
X_both = vect_both.fit_transform(data['Clean Text n-gram'])
X_features_both = pd.DataFrame(X_both.toarray())
X_features_both.columns = vect_both.get_feature_names()

#%%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

#%%
'''
Created bar plots to examine the distribution of the tf-idf scores for each set of n-grams

'''
X_features_scoredf = pd.DataFrame()
X_features_scoredf['TF-IDF Score'] = X_features_uni.apply( lambda col: col.mean(), axis = 0)
X_features_scoredf = X_features_scoredf.sort_values(by=['TF-IDF Score'], ascending=False)
X_features_scoredf = X_features_scoredf.reset_index()

#%%
X_features_scoredf_samp = X_features_scoredf[0:30]
plt.xticks(rotation = 45)
plt.bar(X_features_scoredf_samp['index'], X_features_scoredf_samp['TF-IDF Score'])

#%%
X_features_bi_scoredf = pd.DataFrame()
X_features_bi_scoredf['TF-IDF Score'] = X_features_bi.apply( lambda col: col.mean(), axis = 0)
X_features_bi_scoredf = X_features_bi_scoredf.sort_values(by=['TF-IDF Score'], ascending=False)
X_features_bi_scoredf = X_features_bi_scoredf.reset_index()

#%%
X_features_bi_scoredf_samp = X_features_bi_scoredf[0:30]
plt.xticks(rotation = 65)
plt.bar(X_features_bi_scoredf_samp['index'], X_features_bi_scoredf_samp['TF-IDF Score'])

#%%
X_features_both_scoredf = pd.DataFrame()
X_features_both_scoredf['TF-IDF Score'] = X_features_both.apply( lambda col: col.mean(), axis = 0)
X_features_both_scoredf = X_features_both_scoredf.sort_values(by=['TF-IDF Score'], ascending=False)
X_features_both_scoredf = X_features_both_scoredf.reset_index()

#%%
X_features_both_scoredf_samp = X_features_both_scoredf[0:30]
plt.xticks(rotation = 65)
plt.bar(X_features_both_scoredf_samp['index'], X_features_both_scoredf_samp['TF-IDF Score'])








