#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:20:02 2020

@author: Jingnan
"""
import pandas as pd
from UtilWordEmbedding import DocPreprocess
from gensim.models import word2vec
#import spacy
import spacy_streamlit
from UtilWordEmbedding import TfidfEmbeddingVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor

#---------------------------------
# Set up headers
#st.title("SHORTLISTMe")


st.markdown("<h1 style='text-align: left; color: orange;'>How Much I'LL Get</h1>", unsafe_allow_html=True)


st.write("")
#---------------------------------

#---------------------------------
# Input Box
#---------------------------------


job_title = st.text_input("Job Title")

job_des = st.text_area("Job Description")

job_city = st.text_input("Job Location (City)")

job_state = st.text_input("Job Location (State)")

run = st.button("Calculate!")


#---------------------------------
# Load the data
#---------------------------------
data = pd.read_csv("Cleaned_Data.csv")




#---------------------------------
# Preprocess the data
#---------------------------------
#nlp = spacy.load('en_core_web_md')
nlp = ["en_core_web_sm", "/path/to/model"]
stop_words = spacy.lang.en.stop_words.STOP_WORDS
all_docs = DocPreprocess(nlp, stop_words, data['Job Description'], data['Job Title'])






#---------------------------------
# Word2Vec and Doc2Vec
#---------------------------------
num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count
num_workers = 1       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

w2v_model = word2vec.Word2Vec(all_docs.doc_words, workers=num_workers, \
            size=num_features, min_count = min_word_count,window = context, sample = downsampling)

tfidf_w2v = TfidfEmbeddingVectorizer(w2v_model)
tfidf_w2v.fit(all_docs.doc_words)  # fit tfidf model first
tfidf_doc2vec = tfidf_w2v.transform(all_docs.doc_words)



#---------------------------------
# IV and DV
#---------------------------------
vec = DictVectorizer()
location_title = vec.fit_transform(data[['State', 'City', 'Job Title']].to_dict('records'))

X = hstack([tfidf_doc2vec,location_title])
y_low= data['Low']
y_up = data['High']



#---------------------------------
# Train Random Forest Model
#---------------------------------

rf_low = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_low.fit(X, y_low)

rf_high = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_high.fit(X, y_up)



#---------------------------------
# Predict based on user input
#---------------------------------

if run:
    job_des = [job_des]
    job_title = [job_title]
    job_city = [job_city]
    job_state = [job_state]
    input_doc = DocPreprocess(nlp, stop_words,job_des, job_title)
    tfidf_w2v_input = TfidfEmbeddingVectorizer(w2v_model)
    tfidf_w2v.fit(input_doc.doc_words)  # fit tfidf model first
    tfidf_doc2vec_input = tfidf_w2v.transform(input_doc.doc_words)
    pd_input = pd.DataFrame({'state':job_state,
                             'city':job_city,
                             'title':job_title})
    location_title_input = vec.transform(pd_input.to_dict('records'))
    X_input = hstack([tfidf_doc2vec_input, location_title_input])
    pred_input_low = rf_low.predict(X_input)
    pred_input_high = rf_high.predict(X_input)
    st.header("Your estimated maximum salary is USD {}".format(int(pred_input_high)))
    st.subheader("Your range of prediction is USD {} - USD {}".format(int(pred_input_low), int(pred_input_high)))
else:
    st.write("Waiting calculation!")




