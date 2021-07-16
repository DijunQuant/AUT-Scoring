#!/usr/bin/env python
# coding: utf-8

# # Fluency Methods

# ## Algorithm to Automate Fluency Scoring

# ### Import Packages

# In[1]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string

import shared_functions as sf
from shared_functions import *


# In[ ]:





# ## Fluency Algo 
# ### counting rows belonging to a participant

# In[2]:


# get fluency scores
def get_fluency_score(fluency_rating_df):
    # get id list
    id_list = sf.get_id_list(fluency_rating_df)
    # create fluency dictionary
    participants_fluency = {k: 0 for k in id_list}
    
    # calculate fluency for each participant
    # store fluency in dictionary
    for participant in id_list:
        temp_df = fluency_rating_df.loc[fluency_rating_df['id'] == participant]
        participants_fluency[participant] = len(temp_df.index)
        
    # create fluency df
    fluency_score_df = pd.DataFrame(participants_fluency.items(), columns=['id', 'fluency'])
    
    # return fluency df
    return fluency_score_df


# In[3]:


# return fluency df
def get_fluency(df, stopwords_list, join_list):
    fluency_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
                
    # get fluency df
    fluency_score_df = get_fluency_score(fluency_rating_df)
        
    return fluency_score_df


# In[ ]:





# ## Collect the Method Results

# In[4]:


def save_fluency_scores(data_dict):
    # store the fluency result dataframes for all prompts
    fluency_results_list = []
    # get list of prompts from the data_dict
    data_keys = list(data_dict.keys())
    # iterate through the results list, appending the corresponding fluency table
    for data in data_keys:
        fluency_results_list.append(get_fluency(data_dict[data], sf.stopwords_edited, True))
        
    # return list of fluency df results
    return fluency_results_list


# In[ ]:





# ## Write Fluency Results to CSVs

# In[5]:


underscore = "_"


# In[6]:


# write out the flexibility results into csvs
def write_fluency_results(data_dict, flex_results_list, date):
    # get list of prompts from the data_dict
    data_keys = list(data_dict.keys())
    # iterate through the results list, write out the corresponding freqs table
    for i in range(len(data_keys)):
        flex_results_list[i].to_csv("fluency_results_" + date + underscore + data_keys[i] + ".csv", encoding = 'utf-8', index=False)


# In[ ]:




