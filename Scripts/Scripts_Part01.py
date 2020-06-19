#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jingnan
"""

import numpy as np
import pandas as pd


#load the data
data = pd.read_csv("glassdoor_jobs.csv")




# recode salary and location
data[['Low','High_A']] = data['Salary Estimate'].str.split('-', expand=True)
data[['High','Drop']] = data['High_A'].str.split('(', expand=True)
data = data.drop(columns = ['Drop', 'High_A'])
data['Low'] = data['Low'].replace({'\$':''}, regex = True)
data['High'] = data['High'].replace({'\$':''}, regex = True)
data['Low'] = data['Low'].replace({17:35000})
data['High'] = data['High'].replace({' 23 Per Hour':'50K'})
data['Low'] =data['Low'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int)
data['High'] =data['High'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int)
data['State'] = data['Location'].str.strip().str[-2:]
data['City'] = data['Location'].str.strip().str[:-4]

data.head(10)

# removal null in salary

clean_data = data[['Job Title','Job Description', 'Low', 'High', 'City', 'State']]
clean_data.to_csv('Cleaned_Data.csv')
