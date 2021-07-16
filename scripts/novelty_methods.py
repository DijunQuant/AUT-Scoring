#!/usr/bin/env python
# coding: utf-8

# # Novelty Methods

# ## Algorithm to Automate Novelty Scoring

# ### Import Packages

# In[1]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from nltk.cluster.kmeans import KMeansClusterer

import shared_functions as sf
from shared_functions import *


# In[ ]:





# ## Semantic Spaces
# ### Word2Vec Models for Embeddings

# In[2]:


# load pretrained model
word_model_twitter25 = api.load("glove-twitter-25")


# In[3]:


# create dictionary of counts for each word in model
twitter25_dict = {}
for i in range(len(word_model_twitter25)):
    twitter25_dict[word_model_twitter25.index_to_key[i]] = word_model_twitter25.key_to_index[word_model_twitter25.index_to_key[i]]


# In[4]:


# get the frequency of each word in dictionary
total_words = 0
for key in twitter25_dict:
    total_words = total_words + twitter25_dict[key]
    
for key in twitter25_dict:
    twitter25_dict[key] = twitter25_dict[key]/total_words


# In[ ]:





# ## Novelty Algo 1
# ### Avg Distance using Word2Vec
# ### Lesser Average Similarity, More Novel
# ### More of a Proof of Concept Method

# In[5]:


# method to get the novelty rating
# average of the similarities seen
def get_similarity_word2vec_avg(prompt, phrase_list, word_model):
    avg_sim = 0
    # find similarity of each word in phrase with the prompt
    for term in range(len(phrase_list)):
        avg_sim = avg_sim + word_model.similarity(w1 = prompt, w2 = phrase_list[term])
        
    # take the average
    if len(phrase_list) == 0:
        avg_sim = 0
    else:
        avg_sim = avg_sim/len(phrase_list)
        
    return (avg_sim)


# In[6]:


# method that returns the dataframe with novelty rating for each phrase
def get_novelty_word2vec_avg(df, prompt, stopwords_list, word_model, join_list):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # list to keep parallel list of the responses similarity
    avg_sim_list = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        avg_sim_list.append(get_similarity_word2vec_avg(prompt, response, word_model))

    # add novelty rating list to dataframe
    novel_rating_df['avg_sim'] = avg_sim_list
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 2
# ### Word2Vec + Smooth Inverse Frequency + Cosine Similarity
# 
# ### Does work well because phrases are super short
# ### Don't use
# ### Fix if have time

# In[7]:


# method to apply SIF to the vectors 
def get_sif_feature_vectors(prompt, response, word_model):
    # set the word count dictionary with frequencies
    word_counts = twitter25_dict
    # size of vectore in word embeddings
    embedding_size = 25 
    # set hyper parameter
    a = 0.001
    # list to store vectors
    phrase_set = []
    for phrase in [prompt, response]:
        # zero out the vector
        vs = np.zeros(embedding_size)
        phrase_length = len(phrase)
        for word in phrase:
            # smooth inverse frequency, SIF
            a_value = a / (a + word_counts[word]) 
            # vs += sif * word_vector
            vs = np.add(vs, np.multiply(a_value, word_model[word]))
        # weighted average
        if phrase_length == 0:
            vs[:] = 0
        else:
            vs = np.divide(vs, phrase_length) 
        phrase_set.append(vs)
    # return the SIF adjusted vectors
    return phrase_set


# In[8]:


def get_similarity_word2vec_sif(prompt, response, word_model):
    # get the SIF adjusted vectors
    vectors = get_sif_feature_vectors(prompt, response, word_model)
    # return the cosine similarity
    return (sf.get_cosine_distance(vectors[0], vectors[1]))


# In[9]:


def get_novelty_word2vec_sif(df, prompt, stopwords_list, word_model, join_list):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # list to keep parallel list of the responses similarity
    avg_sim_list = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        avg_sim_list.append(get_similarity_word2vec_sif_cosinesim(prompt, response, word_model))

    # add novelty rating list to dataframe
    novel_rating_df['SIF + cosine sim'] = avg_sim_list
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 3
# ### sem_space + element wise multiplication + cosine distance
# ### Greater Cos Distance, Greater Novelty
# ### Most similar to SemDis

# In[10]:


# method to get the element wise multiplied vector
# multiply vectors in phrase
def get_ew_multiplied_vector(phrase_list, sem_space):
    vectors_list = []
    # add vectors to list
    # change to numpy array
    for term in phrase_list:
        vectors_list.append(np.array(sem_space.loc[term].values.tolist()))
    
    # get element wise multiplied vector
    element_wise_multiplied_vector = np.ones(len(sem_space.columns))

    # calculate element wise multiplied vector
    for vector in vectors_list:
        element_wise_multiplied_vector = element_wise_multiplied_vector * vector

    # return element wise multiplied vector
    return element_wise_multiplied_vector


# In[11]:


# get cosine sim from prompt and ewm
def get_ewm_cosdist(prompt, response, sem_space):
    # get prompt vector and ewm vector using semantic space
    prompt_vector = np.array(sem_space.loc[prompt].values.tolist())
    ewm_vector = get_ew_multiplied_vector(response, sem_space)

    # return cosine dist between prompt and ewm
    return (sf.get_cosine_distance(prompt_vector, ewm_vector))


# In[12]:


# get df with results of the cosine distance from prompt using the elementwise multiplied vectors in the response
def get_novelty_ewm(df, prompt, stopwords_list, sem_space, join_list):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # list to store cosine sims
    cosine_sim_list = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        cosine_sim_list.append(get_ewm_cosdist(prompt, response, sem_space))

    # add novelty rating list to dataframe
    novel_rating_df['ewm_vector_cosine_dis'] = cosine_sim_list
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 4
# ### sem_space + local minina + cosine distance

# In[13]:


# get word in phrase that has the least distance from the prompt
def get_minvec(prompt, phrase_list, sem_space):
    distances_list = []
    # get prompt vector
    prompt_vector = np.array(sem_space.loc[prompt].values.tolist())
    
    # create list of cosine distances
    for term in phrase_list:
        distances_list.append(sf.get_cosine_distance(prompt_vector, np.array(sem_space.loc[term].values.tolist())))
        
    # return the max cosine distance
    return max(distances_list, default=0)


# In[14]:


# get df with results of the cosine distance from prompt using the minima vector in the response
def get_novelty_minvec(df, prompt, stopwords_list, sem_space, join_list):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # list to store cosine sims
    cosine_sim_list = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        cosine_sim_list.append(get_minvec(prompt, response, sem_space))

    # add novelty rating list to dataframe
    novel_rating_df['minima_vector_cosine_dis'] = cosine_sim_list
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 5
# ### element wise multiplication + cosine distance + clustering
# ### average responses cosine distance in the same cluster
# ### idea is that phrases with same alternate task will group
# ### variation in phrase in the same cluster will  be averaged out

# In[15]:


# averages the distance of the phrases in each cluster
# gives each phrase in cluster the average distance 
def get_clustered_novelty_score(novel_rating_df, column, num_clusters, display_clusters):
    # get cluster df
    clusters_df = sf.get_counts_vector(num_clusters, novel_rating_df['response_processed_phrase'].tolist(), display_clusters)
    # get cleaned phrases and their current novelty rating
    novelty_scores = dict(zip(novel_rating_df.response_processed_phrase, novel_rating_df[column]))

    # create dictionary out of cluster df
    clusters = dict(zip(clusters_df.category, clusters_df.responses))
        
    # initialize empty dictionary to store the score for a category
    clusters_scores = dict.fromkeys(clusters)
    
    # get the average cosine distance for a cluster
    for key in clusters:
        # find total of distance of terms in phrase
        score = 0
        for phrase in clusters[key]:
            score = score + novelty_scores[phrase]
        # find the average
        score = score/len(clusters[key])
        # store cluster score in dictionary
        clusters_scores[key] = score
        
    # create dictionary to store a phrase and its new novelty score 
    # new score is the average of the responses in one cluster
    phrase_scores_dict = {}
    for key in clusters:
        for phrase in clusters[key]:
            phrase_scores_dict[phrase] = clusters_scores[key]
            
    # make a list that matches the one in the current dataframe
    # return list to be added to dataframe
    df_phrases_scores_list = [] 
    for phrase in novel_rating_df['response_processed_phrase'].tolist():
        df_phrases_scores_list.append(phrase_scores_dict[phrase])

    return list(df_phrases_scores_list)


# In[16]:


# get df with results of the cosine distance from prompt using the elementwise multiplied vectors in the response
def get_novelty_ewm_cluster(df, prompt, stopwords_list, sem_space, join_list, num_clusters, display_clusters):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # add column for complete phrase
    novel_rating_df['response_processed_phrase'] = [' '.join(x) for x in cleaned_responses]
    # list to store cosine sims
    cosine_sim_list = []
    
    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        print(response)
        cosine_sim_list.append(get_ewm_cosdist(prompt, response, sem_space))

    # add novelty rating list to dataframe
    novel_rating_df['ewm_vector_cosine_dis'] = cosine_sim_list
    
    # add novelty rating for the average rating of a cluster
    novel_rating_df['ewm_vector_cosine_dis_clus_avg'] = get_clustered_novelty_score(novel_rating_df, 'ewm_vector_cosine_dis', num_clusters, display_clusters)
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 6
# ### sem_space + local minima + cosine distance + clustering
# ### average responses cosine distance in the same cluster
# ### idea is that phrases with same alternate task will group
# ### variation in phrase in the same cluster will  be averaged out
# ### differs from algo 5, does local minima not ewm

# In[17]:


# get df with results of the cosine distance from prompt using the elementwise multiplied vectors in the response
def get_novelty_minvec_cluster(df, prompt, stopwords_list, sem_space, join_list, num_clusters, display_clusters):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # add column for complete phrase
    novel_rating_df['response_processed_phrase'] = [' '.join(x) for x in cleaned_responses]
    # list to store cosine sims
    cosine_sim_list = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        cosine_sim_list.append(get_minvec(prompt, response, sem_space))

    # add novelty rating list to dataframe
    novel_rating_df['minima_vector_cosine_dis'] = cosine_sim_list
        
    # add novelty rating for the average rating of a cluster
    novel_rating_df['minima_vector_cosine_dis_clus_avg'] = get_clustered_novelty_score(novel_rating_df, 'minima_vector_cosine_dis', num_clusters, display_clusters)
    
    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Novelty Algo 7
# ### uses the cosine + minima + clustering methods
# ### uses the same clusterr for all algos
# ### two different scoring systems, average or minimum

# In[18]:


# averages the distance of the phrases in each cluster
# gives each phrase in cluster the average distance
# generalized to avg or min, both ewm and minima
def get_clustered_novelty_score_generalized(clusters_df, novel_rating_df, average, column):
    # get cleaned phrases and their current novelty rating
    novelty_scores = dict(zip(novel_rating_df.response_processed_phrase, novel_rating_df[column]))

    # create dictionary out of cluster df
    clusters = dict(zip(clusters_df.category, clusters_df.responses))
        
    # initialize empty dictionary to store the score for a category
    clusters_scores = dict.fromkeys(clusters)
    
    # get the average or min cosine distance for a cluster
    # calculate average
    if average:
        for key in clusters:
            # keep track of total of distances
            score = 0
            for phrase in clusters[key]:
                score = score + novelty_scores[phrase]
            # calculate the average
            score = score/len(clusters[key])
            # store average distance for the cluster
            clusters_scores[key] = score
    # find minimum distance
    else:
        for key in clusters:
            scores_list = []
            # find the cosine distances of phrases in a cluster
            for phrase in clusters[key]:
                scores_list.append(novelty_scores[phrase])
            # store minimum distance as score for the clusterr
            clusters_scores[key] = min(scores_list)
        
    # create dictionary to store a phrase and its new novelty score 
    # new score is the average of the responses in one cluster
    phrase_scores_dict = {}
    for key in clusters:
        for phrase in clusters[key]:
            phrase_scores_dict[phrase] = clusters_scores[key]
            
    # make a list that matches the one in the current dataframe
    # return list to be added to dataframe
    df_phrases_scores_list = [] 
    for phrase in novel_rating_df['response_processed_phrase'].tolist():
        df_phrases_scores_list.append(phrase_scores_dict[phrase])
    
    # uncomment to show clusters df
#     display(clusters_df)
            
    # return scores list thats parallel to the responses column
    return list(df_phrases_scores_list)


# In[19]:


# get df with results of the cosine distance from prompt using the elementwise multiplied vectors in the response
def get_novelty_combined(df, prompt, stopwords_list, sem_space, join_list, num_clusters, display_clusters):
    # clean the responses
    novel_rating_df = sf.get_cleaned_responses_df(df, stopwords_list, join_list)
    cleaned_responses = novel_rating_df['response_processed'].tolist()
    # store combined phrases as a new column
    novel_rating_df['response_processed_phrase'] = [' '.join(x) for x in cleaned_responses]
    # list to store cosine sims for ewm
    cosine_sim_list_ewm = []
    # list to store cosine sims for minima
    cosine_sim_list_minima = []

    # implement algo
    # pass in clean responses
    for response in cleaned_responses:
        # add novelty rating to list 
        cosine_sim_list_ewm.append(get_ewm_cosdist(prompt, response, sem_space))
        cosine_sim_list_minima.append(get_minvec(prompt, response, sem_space))
        
     # get clusters for the dataset
    # idea is to use the same clusters for each analysis
    clusters_df = sf.get_counts_vector(num_clusters, novel_rating_df['response_processed_phrase'].tolist(), display_clusters)

    # add novelty rating list to dataframe for ewm
    novel_rating_df['ewm_vector_cosine_dis'] = cosine_sim_list_ewm
    
    # add the columns for the novelty scores
    novel_rating_df['ewm_vector_cosine_dis_clus_avg'] = get_clustered_novelty_score_generalized(clusters_df, novel_rating_df, True, 'ewm_vector_cosine_dis')

    # add novelty rating list to dataframe for minima
    novel_rating_df['minima_vector_cosine_dis'] = cosine_sim_list_minima
    novel_rating_df['minima_vector_cosine_dis_clus_avg'] = get_clustered_novelty_score_generalized(clusters_df, novel_rating_df, True, 'minima_vector_cosine_dis')
    novel_rating_df['minima_vector_cosine_dis_clus_min'] = get_clustered_novelty_score_generalized(clusters_df, novel_rating_df, False, 'minima_vector_cosine_dis')

    # new column with novelty rating
    return novel_rating_df


# In[ ]:





# ## Write Novelty Results into CSVs

# In[20]:


underscore = "_"


# In[21]:


def write_novelty_results(data_dict, semspace_dict, num_clusters):
    # create list for the different prompts and semantic spaces
    # iterate through each combination of prompt and semantic space
    data_keys = list(data_dict.keys())
    semspace_keys = list(semspace_dict.keys())
    for semspace in semspace_keys:
        for data in data_keys:
            # get the results df for the corresponding combination of prompt and semantic space
            df = get_novelty_combined(data_dict[data], data, stopwords_edited, semspace_dict[semspace], False, num_clusters, False)
            df['novelty_m'] = df[['novelty_1', 'novelty_2']].mean(axis=1)
            # reorder cols the way I want them 
            df = df[['id',
                 'response',
                 'response_nofill',
                 'response_processed',
                 'response_processed_phrase',
                 'item',
                 'item_nofill',
                 'SemDis_factor',
                 'SemDis_cbowukwacsubtitle_nf_m',
                 'SemDis_cbowsubtitle_nf_m',
                 'SemDis_cbowBNCwikiukwac_nf_m',
                 'SemDis_TASA_nf_m',
                 'SemDis_glove_nf_m',
                 'SemDis_MEAN',
                 'ewm_vector_cosine_dis',
                 'ewm_vector_cosine_dis_clus_avg',
                 'minima_vector_cosine_dis',
                 'minima_vector_cosine_dis_clus_avg',
                 'minima_vector_cosine_dis_clus_min',
                 'novelty_1',
                 'novelty_2',
                 'novelty_m']]
            # write results table into csv
            df.to_csv(data + underscore + semspace + underscore + "results" + ".csv", index=False) 


# In[ ]:




