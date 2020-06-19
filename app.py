#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:20:02 2020

@author: Jingnan
"""
import pandas as pd
from UtilWordEmbedding import DocPreprocess
from gensim.models import word2vec
import spacy
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
# Model
#---------------------------------
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
            size=num_features, min_count = min_word_count,window = context, sample = downsampling)

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


#data['avg_salary'] = (data['Low'] + data['High'])/2
#data['Salary_C'] = pd.cut(data['avg_salary'], range(20000, 210000, 20000))
#data['code'] = data.Salary_C.cat.codes
#recode
#data['code_new'] = data['code']+1
#data.loc[(data['code'] == 8),'code_new'] = 8


X = hstack([tfidf_doc2vec,location_title])
y_low= data['Low']
y_up = data['High']
#y_c = data['code_new']
#y_m = data['avg_salary']





### fit the model



rf_low = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_low.fit(X, y_low)


rf_high = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_high.fit(X, y_up)



#x_train, x_test, y_train, y_test = train_test_split(X, y_low, test_size = 0.3)


## Randome Forest

# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
#rf.fit(x_train, y_train)
# Use the forest's predict method on the test data
#predictions = rf.predict(x_test)
# Calculate the absolute errors
#errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
#mape = 100 * (errors / y_test)
# Calculate and display accuracy
#accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')


#####prediction


########

#job_des = ["Position Description: This position is responsible for business consulting activities for the Data Strategy and Analytics teams within our client organizations to monitor and assist in improving their analytics eco system. We need someone with a creative problem-solving skills to work on our client's business opportunities. Our Data Scientistwill work to understand high value opportunities and identify/build potential solutions, which often involve discovering new insights by transforming data into intuitive & interactive visualizations or applications Responsibilities: Serve as an expert in translating complex data into key strategy insights and valuable actions. * Discover business narratives told by the data and present them to other scientists, business stakeholders, and managers at various levels. * Develop and test heuristics. * Create and run statistic/ML models. * Perform data exploration and data mining. * Create business intelligence, dashboards, visualizations, and/or other advanced analytics reports to adequately tell the business narrative and offer recommendations that are practiced, actionable, and have material impact, in addition to being well-supported by analytical models and data. Skills: Work experience in addition to degree: 3 - 5 years of data engineering/science related activities with overall 10+ years' experience. * Graduation from a four-year college or university with a degree in statistics, physics, mathematics, engineering, computer science, or management of information systems. * Expert knowledge of SAS, Python Machine Learning or R * Working knowledge of statistics, programming and predictive modeling. * Working knowledge of code writing * Big Data/Hadoop/NoSQL and experience with large datasets Desired: Doctorate in Computer science, Engineering, Physics or Statistics Skills: * Data Analysis * SQL * Python * QlikView * Data Architecture What you can expect from us: Build your career with us. It is an extraordinary time to be in business. As digital transformation continues to accelerate, CGI is at the center of this change-supporting our clients' digital journeys and offering our professionals exciting career opportunities. At CGI, our success comes from the talent and commitment of our professionals. As one team, we share the challenges and rewards that come from growing our company, which reinforces our culture of ownership. All of our professionals benefit from the value we collectively create. Be part of building one of the largest independent technology and business services firms in the world. Learn more about CGI at www.cgi.com. No unsolicited agency referrals please. CGI is an equal opportunity employer. Qualified applicants will receive consideration for employment without regard to their race, ethnicity, ancestry, color, sex, religion, creed, age, national origin, citizenship status, disability, medical condition, military and veteran status, marital status, sexual orientation or perceived sexual orientation, gender, gender identity, and gender expression, familial status, political affiliation, genetic information, or any other legally protected status or characteristics. CGI provides reasonable accommodations to qualified individuals with disabilities. If you need an accommodation to apply for a job in the U.S., please email the CGI U.S. Employment Compliance mailbox at US_Employment_Compliance@cgi.com. You will need to reference the requisition number of the position in which you are interested. Your message will be routed to the appropriate recruiter who will assist you. Please note, this email address is only to be used for those individuals who need an accommodation to apply for a job. Emails for any other reason or those that do not include a requisition number will not be returned. We make it easy to translate military experience and skills! Click here to be directed to our site that is dedicated to veterans and transitioning service members. All CGI offers of employment in the U.S. are contingent upon the ability to successfully complete a background investigation. Background investigation components can vary dependent upon specific assignment and/or level of US government security clearance held. CGI will not discharge or in any other manner discriminate against employees or applicants because they have inquired about, discussed, or disclosed their own pay or the pay of another employee or applicant. However, employees who have access to the compensation information of other employees or applicants as a part of their essential job functions cannot disclose the pay of other employees or applicants to individuals who do not otherwise have access to compensation information, unless the disclosure is (a) in response to a formal complaint or charge, (b) in furtherance of an investigation, proceeding, hearing, or action, including an investigation conducted by the employer, or (c) consistent with CGI's legal duty to furnish information"]
#job_title = ["Data Scientist"]
#job_city = ['San Francisco']
#job_state = ['CA']







#---------------------------------
# Output Box
#---------------------------------


#checking prediction house price
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
    st.subheader("Your range of prediction is USD {} - USD {}".format(int(pred_input_low,pred_input_high)))
else:
    st.write("Waiting calculation!")




