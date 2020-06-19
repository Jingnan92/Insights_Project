#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jingnan
"""

import pandas as pd
from UtilWordEmbedding import DocPreprocess
from gensim.models import word2vec
import spacy
from UtilWordEmbedding import TfidfEmbeddingVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


data = pd.read_csv("Cleaned_Data.csv")

# pre-process the job description
nlp = spacy.load('en_core_web_md')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
all_docs = DocPreprocess(nlp, stop_words, data['Job Description'], data['Job Title'])



# train word2vec model

num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count
num_workers = 1       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

w2v_model = word2vec.Word2Vec(all_docs.doc_words, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model_name = "JobDes"
w2v_model.save(model_name)
w2v_model.wv.most_similar("programming")


#TF-IDF weighted Word Embeddings

tfidf_w2v = TfidfEmbeddingVectorizer(w2v_model)
tfidf_w2v.fit(all_docs.doc_words)  # fit tfidf model first
tfidf_doc2vec = tfidf_w2v.transform(all_docs.doc_words)


# Transform text data on other variables

vec = DictVectorizer()
location_title = vec.fit_transform(data[['State', 'City', 'Job Title']].to_dict('records'))


# Prepare IV and DV

X = hstack([tfidf_doc2vec,location_title])
y_min= data['Low']
y_max = data['High']


### split into train and test

x_train_min, x_test_min, y_train_min, y_test_min = train_test_split(X, y_min, test_size = 0.3)
x_train_max, x_test_max, y_train_max, y_test_max = train_test_split(X, y_max, test_size = 0.3)



## Randome Forest Models: one for maximum wage, one for minimum wage

## Model I: fit and predict
rf_min = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_min.fit(x_train_min, y_train_min)
predictions_min = rf_min.predict(x_test_min)


## Model I: evaluation 
errors_min = abs(predictions_min - y_test_min)
mape_min = 100 * (errors_min / y_test_min)
accuracy_min = 100 - np.mean(mape_min)
print('Accuracy:', round(accuracy_min, 2), '%.')


## Model II: fit and train
rf_max = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_max.fit(x_train_max, y_train_max)
predictions_max = rf_min.predict(x_test_max)

## Model II: evaluation
errors_max = abs(predictions_max - y_test_max)
mape_max = 100 * (errors_max / y_test_max)
accuracy_max = 100 - np.mean(mape_max)
print('Accuracy:', round(accuracy_max, 2), '%.')





