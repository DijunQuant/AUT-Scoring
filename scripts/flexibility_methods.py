#!/usr/bin/env python
# coding: utf-8

# # Flexibility Methods

# ## Algorithm to Automate Flexibility Scoring

# ### Import Packages

# In[1]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import TfidfVectorizer

from yellowbrick.cluster import KElbowVisualizer

from nltk.cluster.kmeans import KMeansClusterer

import shared_functions as sf
from shared_functions import *


# In[ ]:





# ## Flexibility Algo 
# ### tf-idf scikit-learn + clustering

# In[2]:


def get_flexibility_score(flexibility_rating_df, num_clusters, responses, display_clusters):
    clusters_df = sf.get_tfidf_vector(num_clusters, responses, display_clusters)
    
    # create dictionary out of cluster df
    # has clusters and their respective responses
    clusters = dict(zip(clusters_df.category, clusters_df.responses))
    
    flex_df_cleaned = flexibility_rating_df[flexibility_rating_df.response_processed != '']
    participants = get_id_list(flex_df_cleaned)
    participants_responses_list = list(zip(flex_df_cleaned.id, flex_df_cleaned.response_processed))
    
    # get dictionary of each participants responses
    participants_responses_dict = {k: [] for k in participants}
    
    for index in range(len(participants_responses_list)):
        participants_responses_dict[participants_responses_list[index][0]].append(participants_responses_list[index][1])
        
    # get dictionary of responses and their respective dictionary
    responses_cluster_rep = {}
    
    for key in clusters:
        for phrase in clusters[key]:
            responses_cluster_rep[phrase] = key
            
    # get dictionary of participants and clusters their responses existed in
    participants_clusters_apperance = {k: [] for k in participants}
    
    for index in range(len(participants_responses_list)):
        participants_clusters_apperance[participants_responses_list[index][0]].append(responses_cluster_rep[participants_responses_list[index][1]])
        
    # get dic of number of clusters a participants responses are in
    
    participants_clusters_seen = {k: [] for k in participants}
    
    for participant in participants_clusters_seen:
        responses_set = set(participants_clusters_apperance[participant])
        participants_clusters_seen[participant] = len(responses_set)
    
#     print(clusters)
#     print()
#     print(participants_responses_list)
#     print()
#     print(participants_responses_dict)
#     print()
#     print(responses_cluster_rep)
#     print()
#     print(participants_clusters_apperance)
#     print()
#     print(participants_clusters_seen)
    
    # create flexiblity df
    flexibility_df = pd.DataFrame(participants_clusters_seen.items(), columns=['id', 'flexibility'])
    return flexibility_df


# In[3]:


# get the flexiblity score using tfidf and clustering
def get_flexibility_tfidf_scikit_learn_clustering(df, stopwords_list, num_clusters, join_list, display_clusters):
    # get cleaned df
    flexibility_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    # get phrases into a list
    responses = flexibility_rating_df['response_processed'].tolist()
                
    # add flexibility df
    flexibility_rating_df = get_flexibility_score(flexibility_rating_df, num_clusters, responses, display_clusters)
        
    return flexibility_rating_df


# In[ ]:





# ## Calculate Method Results

# In[4]:


# print flexibility results for human rater and method
def print_flexibility_scores(data_dict, num_clusters):
    # store the flexibility result dataframes for all prompts
    flexibility_df_list = []
    # list of the keys in the data dictionary passed in
    data_keys = list(data_dict.keys())
    for data in data_keys:
        # get list of id
        id_list = sf.get_id_list(data_dict[data])
        participants_clusters_seen = []
        # find the unique instance in each partipants flexibility df
        for participant in id_list:
            # subset dataframe and count unique categories marked
            id_df = data_dict[data][data_dict[data].id == participant]
            flex_1_apperance = len(id_df['flexibility_1'].unique())
            flex_2_apperance = len(id_df['flexibility_2'].unique())
            # store id and flexibility score as tuple
            participants_clusters_seen.append((participant, flex_1_apperance, flex_2_apperance))
        # make df out of tuple
        flexibility_df_rater = pd.DataFrame(participants_clusters_seen, columns=['id', 'flex_1', 'flex_2'])
        # get flexibility score from algo
        flexibility_df_method = get_flexibility_tfidf_scikit_learn_clustering(data_dict[data], stopwords_edited, num_clusters, True, False)
        # merge the dataframes and rename columnsb
        df_cd = pd.merge(flexibility_df_rater, flexibility_df_method, how='inner', on = 'id')
        df_cd.columns = [['id','flex_1', 'flex_2','flex_method']]
        flexibility_df_list.append(df_cd)
     
    # return the df with human and algo flexibility scores
    return flexibility_df_list


# In[5]:


# print out method results meaned, rerun x times
def print_flexibility_scores_avg(data_dict, num_clusters, reruns):
    # store the flexibility result dataframes for all prompts
    flexibility_df_list = []
    # list of the keys in the data dictionary passed in
    data_keys = list(data_dict.keys())
    for data in data_keys:
        # get list of id
        id_list = sf.get_id_list(data_dict[data])
        participants_clusters_seen = []
        # find the unique instance in each partipants flexibility df
        for participant in id_list:
            # subset dataframe and count unique categories marked
            id_df = data_dict[data][data_dict[data].id == participant]
            flex_1_apperance = len(id_df['flexibility_1'].unique())
            flex_2_apperance = len(id_df['flexibility_2'].unique())
            # store id and flexibility score as tuple
            participants_clusters_seen.append((participant, flex_1_apperance, flex_2_apperance))
        # make df out of tuple
        df_cd = pd.DataFrame(participants_clusters_seen, columns=['id', 'flex_1', 'flex_2'])
        # rerun algo to take the average of the results
        for y in range(reruns):
            flexibility_df_method = get_flexibility_tfidf_scikit_learn_clustering(data_dict[data], stopwords_edited, num_clusters, True, False)
            # merge the dataframes
            df_cd = pd.merge(df_cd, flexibility_df_method, how='inner', on = 'id')
        df_cd['flex_method_avg'] = df_cd.iloc[:,3:7].mean(axis=1)
        # rename columns
        df_cd = df_cd[['id','flex_1', 'flex_2','flex_method_avg']]
        flexibility_df_list.append(df_cd)
        
    # return the df with human and algo flexibility scores        
    return flexibility_df_list 


# In[ ]:





# ## Write the Flexibility Results to CSVs

# In[6]:


underscore = "_"


# In[7]:


# write out the flexibility results
def write_flexibility_results(data_dict, flex_results_list, date):
    # get list of prompts from the data_dict
    data_keys = list(data_dict.keys())
    # iterate through the results list, write out the corresponding flexibility table
    for i in range(len(data_keys)):
        flex_results_list[i].to_csv("flexibility_results_" + date + underscore + data_keys[i] + ".csv", encoding = 'utf-8', index=False)


# In[ ]:




