#!/usr/bin/env python
# coding: utf-8

# # Shared Functions between the Different AUT Metric Methods

# ## Imports

# In[1]:


import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string

from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from nltk.cluster.kmeans import KMeansClusterer

from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


global stopwords_edited


# ## Preprocessing Data Functions

# In[3]:


stopwords_edited = list(STOP_WORDS)
stopwords_edited.append("thing")
stopwords_edited.append("use")
stopwords_edited.append("things")


# In[4]:


# method to clean the responses
def process_text(text, stopwords_list, join_list):
    # tokenize text, lemmanize words, removing punctuation, remove stop words, lowercase all words

    # hardcorded for special situations
    text = re.sub("/|-"," ", text)
    text = text.translate(str.maketrans('','',string.punctuation))
    tokens = word_tokenize(text)

    tokens = [w.lower() for w in tokens]
    
    tokens = [word for word in tokens if word not in stopwords_list]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if join_list:
        tokens = ' '.join(tokens)
 
    return tokens


# In[5]:


# method to add a new column
# new column are cleaned responses
def get_cleaned_responses_df(df, stopwords_list, join_list):
    # id_df = df[df.id == id]
    df_processed = df.copy(deep=True)
    responses = df['response'].tolist()

    # make list of processed responses
    for response in range(len(responses)):
        responses[response] = process_text(responses[response], stopwords_list, join_list)

    # add list as column in df
    df_processed['response_processed'] = responses
    
    df_processed = df_processed[df_processed.astype(str)['response_processed'] != '[]']

    return df_processed


# In[ ]:





# ## Working with Dataframe Functions

# In[6]:


# method to get a list of participants
def get_id_list(df):
    id_list = df['id'].unique()
    id_list = sorted(id_list)
    return id_list


# In[ ]:





# ## Calculate Cosine Distance

# In[7]:


# method to calculate cosine distance
def get_cosine_distance(feature_vec_1, feature_vec_2):
    return (1 - cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0])


# In[ ]:





# ## Clustering Function

# In[8]:


# clusters the responses
# get a df of the clusters and their respective phrases
def get_counts_vector(num_clusters, responses, display_clusters):
    # initialize CountVectorizer object
    count_vectorizer = CountVectorizer()
    # vectorize the phrases
    word_count = count_vectorizer.fit_transform(responses)
    
    # elbow method to visualize and find out how many clusters to use
#     visualizer = KElbowVisualizer(KMeans(), k=(10,35), timings=False)
#     visualizer.fit(word_count.toarray())       
#     visualizer.show()

    # nltk kmeans cosine distance implementation
    number_of_clusters = num_clusters
    kmeans = KMeansClusterer(number_of_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
    assigned_clusters = kmeans.cluster(word_count.toarray(), assign_clusters=True)

    # cluster results scikit-learn
    results = pd.DataFrame()
    results['text'] = responses
#     results['category'] = kmeans.labels_
    results['category'] = assigned_clusters
    
    # create dictionary to organize the clusters with their respective phrases
    results_dict = {k: g["text"].tolist() for k,g in results.groupby("category")}
    
    # df of the clusters and the 
    clusters_df = pd.DataFrame(list(results_dict.items()),columns = ['category','responses']) 
    
    if display_clusters:
        display(clusters_df)
    
    return clusters_df


# In[9]:


def get_tfidf_vector(num_clusters, responses, display_clusters):
    # initialize CountVectorizer object
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    # vectorize the phrases
    tfidf = tfidf_vectorizer.fit_transform(responses)
    
    # elbow method to visualize and find out how many clusters to use
#     visualizer = KElbowVisualizer(KMeans(), k=(1,12), timings=False)
#     visualizer.fit(tfidf.toarray())       
#     visualizer.show()

    # nltk kmeans cosine distance implementation
    number_of_clusters = num_clusters
    kmeans = KMeansClusterer(number_of_clusters, repeats=25, distance=nltk.cluster.util.cosine_distance, avoid_empty_clusters=True)
    assigned_clusters = kmeans.cluster(tfidf.toarray(), assign_clusters=True)
        
    # cluster results scikit-learn
    results = pd.DataFrame()
    results['text'] = responses
#     results['category'] = kmeans.labels_
    results['category'] = assigned_clusters
    
    # create dictionary to organize the clusters with their respective phrases
    results_dict = {k: g["text"].tolist() for k,g in results.groupby("category")}
    
    # df of the clusters and the 
    clusters_df = pd.DataFrame(list(results_dict.items()),columns = ['category','responses']) 
    
    if display_clusters:
        display(clusters_df)
    
    return clusters_df


# In[ ]:





# In[ ]:




