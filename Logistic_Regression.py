# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:49:21 2021

@author: jackw
"""

#%% Set Up
# Imports
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

data = pd.read_csv(r"C:\Users\jackw\Desktop\FiCli\Data_final.csv", encoding='unicode_escape')

#%% text clean function
'''
takes text and returns lemmatized text with no punctuation in a tokenized list format
@param text, raw text to be formatted
@return text, formatted text
'''
def clean_text_tfidf(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])  # remove punc
    tokens = re.split('\W+', text)  # split
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]  # lemmatize and tokenize
    return text

'''
takes text and returns lemmatized text with no punctuation in a non-tokenized format
@param text, raw text to be formatted
@return text, formatted text
'''
def clean_text_ngrams(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])  # remove punc
    tokens = re.split('\W+', text)  # split
    text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords])  # lemmatize
    # text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
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
# Set classes
data['Included'] = data['Projected or Documented'].apply(lambda x: set_class(x))

#%%
# Apply cleaning functions
data['Clean Text TFIDF'] = data['Text'].apply(lambda x: clean_text_tfidf(x))
data['Clean Text n-gram'] = data['Text'].apply(lambda x: clean_text_ngrams(x))

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.pipeline import Pipeline

#%%
pipe_logreg = Pipeline([('vect', TfidfVectorizer()),
                        ('fs', SelectKBest()),
                       ('clf', LogisticRegression())])
test_grid = {}
search_grid = {
        'vect__ngram_range': [(1,1), (1,2)],
        'vect__use_idf': (True, False),
        'fs__k': (2000, 5000, 10000, 'all'),
        'clf__penalty': ('l1', 'l2'),
        'clf__C': np.logspace(-4, 4, 20),
        'clf__solver': ['liblinear'],
        'clf__verbose': [0]
        }
#%%
'''
This function is a wrapper function for a model fit using GridSearchCV. The passed data is split into test and training subsets
and then is used to fit the desired model declared in the pipeline object. Each hyperparameter declared in the parameter grid is 
tested and the best hyperparameters will be stored in the grid object. A classification report and confusion matrix are then
created
@param data, pandas dataframe
@param pipeline, scikit pipeline object with vectorizer and model
@param params, dict of hyperparameters to be tested by GridSearchCV
'''
def grid_search(data, pipeline, params):
    # Set path to save plot
    # FILL PATH
    path = r"C:\Users\jackw\Desktop\FiCli\Model Results\{}.png".format(pipeline[2])

    # Test/train set split
    X_train, X_test, y_train, y_test = train_test_split(data['Clean Text n-gram'], data['Included'], test_size=.2)

    # Create grid object and fit to training data
    grid = GridSearchCV(pipeline, params, cv=10)
    grid.fit(X_train, y_train)

    # Use model and create classification report/confusion matrix
    y_pred = grid.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, y_pred, digits=4, output_dict=True)).transpose()
    plot_confusion_matrix(grid, X_test, y_test, display_labels=['Included', 'Not Included'])
    plt.savefig(path)
    print(grid.best_params_)

    return report

#%%
results = grid_search(data, pipe_logreg, search_grid)
#%%
results.to_excel(r"C:\Users\jackw\Desktop\FiCli\Model Results\Logistic_Regression.xlsx")
#%%
LogisticRegression().get_params().keys()