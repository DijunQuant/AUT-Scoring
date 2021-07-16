#!/usr/bin/env python
# coding: utf-8

# # Originality Methods

# ## Algorithm to Automate Originality Scoring

# ### Import Packages

# In[1]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from nltk.cluster.kmeans import KMeansClusterer

import shared_functions as sf
from shared_functions import *


# In[ ]:





# ## Originality Algo 
# ### Counter Vectorizer + clustering

# In[2]:


# calculate originality score for a phrase based on number of responses in a cluster
# calculate the rarity of a phrase
def get_clustered_originality_score(originality_rating_df, num_clusters, responses, display_clusters):
    # get cluster df
    clusters_df = sf.get_counts_vector(num_clusters, responses, display_clusters)
    # create dictionary out of cluster df
    clusters = dict(zip(clusters_df.category, clusters_df.responses))
        
    # initialize empty dictionary to store the score for a category
    clusters_scores = dict.fromkeys(clusters)
    
    # initialize empty dictionary to store 
    # will store 0 or 1, 1 if only 1 response in cluster
    clusters_rarity = dict.fromkeys(clusters)
    
    # get the average cosine distance for a cluster
    for key in clusters:
        score = 1.0
        score = score - (len(clusters[key])/len(originality_rating_df.index))
        clusters_scores[key] = score
        if (len(clusters[key]) == 1):
            clusters_rarity[key] = 1
        else:
            clusters_rarity[key] = 0
        
    # create dictionary to store a phrase and its new originality score 
    # new score is the average of the responses in one cluster
    phrase_scores_dict = {}
    # create dictionary to store a phrase and its rarity 
    # will be 0 or 1, 1 if only response in that cluster
    phrase_rarity_dict = {}
    for key in clusters:
        for phrase in clusters[key]:
            phrase_scores_dict[phrase] = clusters_scores[key]
            phrase_rarity_dict[phrase] = clusters_rarity[key]
            
    # make a list that matches the one in the current dataframe
    # return list to be added to dataframe
    df_phrases_scores_list = [] 
    # make a list that matches the one in the current dataframe
    # return list to be added to dataframe
    df_phrases_rarity_list = []
    for phrase in originality_rating_df['response_processed'].tolist():
        df_phrases_scores_list.append(phrase_scores_dict[phrase])
        df_phrases_rarity_list.append(phrase_rarity_dict[phrase])
                    
    # return the two parallel list of the inverse mapping of the cluster frequency
    # and the rarity marking (1 or 0)
    return (df_phrases_scores_list, df_phrases_rarity_list)


# In[3]:


# create df of each participants originality count score, the amount of 
def get_originality_score_df(originality_df):
    # get id list and create empty dict
    id_list = get_id_list(originality_df)
    participants_originality = {k: 0 for k in id_list}
    # add the number of unique responses aka clusters with only 1 response
    for participant in id_list:
        id_df = originality_df[originality_df.id == participant]
        participants_originality[participant] = id_df['originality'].sum()
    
    # return counts df by participant
    return pd.DataFrame(participants_originality.items(), columns=['id', 'originality'])


# In[4]:


# calculate the originality freq and counts for a response and participant respectively
def get_originality_count_vectorizer(df, stopwords_list, num_clusters, join_list, display_clusters):
    # clean the dataframe
    originality_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    responses = originality_rating_df['response_processed'].tolist()
            
    originality_results = get_clustered_originality_score(originality_rating_df, num_clusters, responses, display_clusters)
    # add frequency column to results df
    originality_rating_df['cluster_freq'] = originality_results[0]
    # add 1/0 column to show if that phrase was by itself in a cluster
    originality_rating_df['originality'] = originality_results[1]
    
    # add the transformed values of the cluster frequency as a column
    originality_rating_df['freq'] = (1 - originality_rating_df['cluster_freq'])
    originality_rating_df['t_freq'] = (.05/(.05 + originality_rating_df['freq']))**2
    
    # get the original phrases counts df
    originality_scores = get_originality_score_df(originality_rating_df)

    # return tuple of freq and count results respectively
    return (originality_rating_df, originality_scores)


# In[ ]:





# ## Collect the Method Results

# In[5]:


# calculate originality scores from method
def get_originality_scores(data_dict, num_clusters):
    # list to store the originality results
    results_list = []
    # list of the keys in the data dictionary passed in
    data_keys = list(data_dict.keys())
    for data in data_keys:
        # get originality results for each dataset
        results = get_originality_count_vectorizer(data_dict[data], sf.stopwords_edited, num_clusters, True, False)
        # add originality results to a list
        results_list.append(results)
        
    # return list
    return results_list


# In[ ]:





# ## Write Originality Results into CSVs

# In[6]:


underscore = "_"


# In[7]:


# write originality freq results into csvs
def write_originality_results_freq(data_dict, results, date):
    # get list of prompts from the data_dict
    data_keys = list(data_dict.keys())
    # iterate through the results list, write out the corresponding freqs table
    for i in range(len(data_keys)):
        results[i][0].to_csv("originality_results_" + date + underscore + "freqs" + underscore + data_keys[i] + ".csv", encoding = 'utf-8', index=False)      


# In[8]:


# write originality counts results into csvs
def write_originality_results_counts(data_dict, results, date):
    # get list of prompts from the data_dict
    data_keys = list(data_dict.keys())
    # iterate through the results list, write out the corresponding counts table
    for i in range(len(data_keys)):
        results[i][1].to_csv("originality_results_" + date + underscore + "counts" + underscore + data_keys[i] + ".csv", encoding = 'utf-8', index=False)    


# In[ ]:




