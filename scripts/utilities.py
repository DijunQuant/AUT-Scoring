import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re

from nltk.stem import WordNetLemmatizer
import string
from IPython.display import display
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import random
import numpy.linalg as la

# stop stopwords list as global variable


# add special cases to stop words list
stopwords_edited = list(STOP_WORDS)
stopwords_edited.append("thing")
stopwords_edited.append("use")
stopwords_edited.append("things")

spNameMap={'cbow_6_ukwac_subtitle':'cbowu',
           'cbow_subtitle':'cbows',
           'ukwac':'cboww',
           'TASA':'tasa',
           'glove_6B':'glov'}

def lemmatize(text, stopwords_list):
    # replace symbols with spaces
    text = re.sub("/|-", " ", text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize the phrase
    tokens = word_tokenize(text)

    # lowercase all tokens
    tokens = [w.lower() for w in tokens]

    # remove stopwords
    tokens = [word for word in tokens if word not in stopwords_list]

    # lemmatize words if needed
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


# method to get the element wise multiplied vector
# multiply vectors in phrase
def get_ew_multiplied_vector(phrase_list, sem_space):
    vectors_list = []
    # add vectors to list
    # change to numpy array
    for term in phrase_list:
        try:
            vectors_list.append(np.array(sem_space.loc[term].values.tolist()))
        except:
            print('cannot find '+term)
    # get element wise multiplied vector
    if len(vectors_list)==0: return np.nan
    element_wise_multiplied_vector = np.ones(len(sem_space.columns))
    # calculate element wise multiplied vector
    for vector in vectors_list:
        element_wise_multiplied_vector = element_wise_multiplied_vector * vector

    # return element wise multiplied vector
    return element_wise_multiplied_vector

# method to calculate cosine distance
def get_cosine_distance(feature_vec_1, feature_vec_2):
    return (1 - cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0])


# get word in phrase that has the least distance from the prompt
def get_minvec(prompt, phrase_list, sem_space):
    distances_list = []
    # get prompt vector
    prompt_vector = np.array(sem_space.loc[prompt].values.tolist())

    # create list of cosine distances
    for term in phrase_list:
        try:
            distances_list.append(get_cosine_distance(prompt_vector, np.array(sem_space.loc[term].values.tolist())))
        except:
            print('cannot find '+term )
    if len(distances_list)==0:
        return np.nan
    else:
        # return the max cosine distance
        return max(distances_list, default=0)
# get word in phrase that has the least distance from the prompt
def get_weightedscore(prompt, phrase_list, sem_space):
    distances_list = []
    # get prompt vector
    prompt_vector = np.array(sem_space.loc[prompt].values.tolist())

    # create list of cosine distances
    for term in phrase_list:
        try:
            distances_list.append(get_cosine_distance(prompt_vector, np.array(sem_space.loc[term].values.tolist())))
        except:
            print('cannot find '+term )
    if len(distances_list)==0:
        return np.nan
    else:
        # return the max cosine distance
        rank=scipy.stats.rankdata(distances_list)
        weight=rank/np.sum(rank)
        return np.sum(np.dot(weight,distances_list))

# get cosine sim from prompt and ewm
def get_ewm_cosdist(prompt, response, sem_space):
    # get prompt vector and ewm vector using semantic space
    prompt_vector = np.array(sem_space.loc[prompt].values.tolist())
    ewm_vector = get_ew_multiplied_vector(response, sem_space)
    if type(ewm_vector)==np.ndarray:
        # return cosine dist between prompt and ewm
        return (get_cosine_distance(prompt_vector, ewm_vector))
    else:
        return np.nan



# clusters the responses
# get a df of the clusters and their respective phrases
#responses are a list of response, each response is a cleaned word list
def cluster_count_vectors(responses, display_clusters,sp=[],repeat=13,max_cluster=99999,min_cluster=30):
    if len(sp)==0:
        # initialize CountVectorizer object
        count_vectorizer = CountVectorizer()
        # vectorize the phrases
        response_vector = count_vectorizer.fit_transform([' '.join(x) for x in responses])
        # elbow method to visualize and find out how many clusters to use
        #     visualizer = KElbowVisualizer(KMeans(), k=(10,35), timings=False)
        #     visualizer.fit(word_count.toarray())
        #     visualizer.show()
        response_vector=response_vector.toarray()
        nanmask=np.zeros(len(responses)).astype(bool)
    else:
        response_vector = [get_ew_multiplied_vector(x, sp) for x in responses]
        nanmask = np.array([type(x)==float for x in response_vector])
        response_vector = np.stack(np.array(response_vector)[~nanmask])

    # nltk kmeans cosine distance implementation
    rng = random.Random()
    rng.seed(123)

    length = np.sqrt((np.array(response_vector) ** 2).sum(axis=1))[:, None]
    #visualizer = KElbowVisualizer(KMeans(), k=(min(min_cluster,int(len(responses)/4)),min(int(len(responses)*0.75),max_cluster)), timings=False)
    fitdata=np.array(response_vector)/length

    #visualizer.fit(fitdata)
    #n_cluster=visualizer.elbow_value_
    m1=min(min_cluster,int(len(responses)/4))
    m2=min(int(len(responses)*0.75),max_cluster)
    n_cluster=m2
    lastsi=0
    for n in range(m1,m2,10):
        kmeans = KMeans(n_clusters=int(n))
        kmeans.fit(fitdata)
        si=silhouette_score(fitdata, kmeans.labels_)
        print(int(n), si, np.quantile(np.bincount(kmeans.labels_), [0.05, 0.1, 0.2]))
        if (si>0.5) | (si<lastsi) | (np.quantile(np.bincount(kmeans.labels_), .1)<2):
            n_cluster=int(n)
            break
        lastsi=si
    print('# of clusters for '+str(len(responses))+' : '+str(n_cluster))

    kmeans = KMeansClusterer(n_cluster, distance=nltk.cluster.util.cosine_distance,
                             repeats=repeat, rng=rng, avoid_empty_clusters=True)

    #visualizer.fit(X)  # Fit the data to the visualizer
    #visualizer.show()

    assigned_clusters = kmeans.cluster(np.array(response_vector), assign_clusters=True)


    # cluster results scikit-learn
    results = pd.DataFrame()
    results['text'] = responses
    #     results['category'] = kmeans.labels_
    results['category']=-1
    results.loc[~nanmask,'category'] = assigned_clusters
    # create dictionary to organize the clusters with their respective phrases
    results_dict = {k: g["text"].tolist() for k, g in results.groupby("category")}
    # df of the clusters and the
    clusters_df = pd.DataFrame(list(results_dict.items()), columns=['category', 'responses'])
    # show clusters df if parameter true
    if display_clusters:
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(clusters_df)
    return results,clusters_df

#compute centroid of the list of vectors, as the average of the normalized vectors
def calc_centroid(vector_list):
    vector_len=len(vector_list[0]) #the length of the embedded vector
    centroid=np.zeros(vector_len)
    for v in vector_list:
        centroid=centroid+v/np.linalg.norm(v)
    centroid=centroid/len(vector_list)
    return centroid

#main function to compute the diversity metric, input is the responses from same participant for a single prompt
#each response is already converted to embedded vectors using certain composition
def calc_diversity(vector_list):
    centroid=calc_centroid(vector_list)
    dist=[]
    for v in vector_list:
        dist.append(get_cosine_distance(centroid,v))
    #return the max of all distance, root mean square of the distance
    #they are equivalent in some sense, root mean squre might behave sligthly better as the diversity metric
    return np.max(dist),np.sqrt(np.mean(np.array(dist)**2))
def dispersion_vectors(responses,sp):
    if len(responses)<2:
        return np.nan,np.nan
    response_vector = [get_ew_multiplied_vector(x, sp) for x in responses]
    response_vector = [x for x in response_vector if type(x)!=float]
    return calc_diversity(response_vector)

def getMinVolEllipse(P, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]

        Returns:
        (center, radii, rotation)

        """
        (N, d) = np.shape(P)
        d = float(d)

        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T

        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT, np.dot(la.inv(V), Q)))  # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse
        center = np.dot(P.T, u)

        # the A matrix for the ellipse
        A = la.inv(
            np.dot(P.T, np.dot(np.diag(u), P)) -
            np.array([[a * b for b in center] for a in center])
        ) / d

        # Get the values we'd like to return
        U, s, rotation = la.svd(A)
        radii = 1.0 / np.sqrt(s)

        #return (center, radii, rotation)
        return np.exp(np.mean(np.log(radii)))


def testCorrDiff(df,colBenchmark,col1,col2):
    r13=df[[col1,colBenchmark]].corr().values[0,1]
    r23=df[[col2,colBenchmark]].corr().values[0,1]
    r12=df[[col1,col2]].corr().values[0,1]
    n=len(df[[col1,col2,colBenchmark]].dropna())
    t_dist = scipy.stats.t(n-3)
    tdif=(r13-r23)*np.sqrt((n-3)*(1+r12)/(2*(1-r13**2-r23**2+2*r13*r23*r12)))
    return tdif,2*(1-t_dist.cdf(np.abs(tdif)))

def printTriNiceCorrTable(data,col):
    rho =data.corr().loc[col,col]
    pval = data.corr(method=lambda x, y: scipy.stats.pearsonr(x, y)[1]).loc[col,col]
    #display(np.round(pval,2))
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
    #display(p)
    newDF=(rho.round(2).astype(str) + p)
    for i in range(len(col)):
        for j in range(i,len(col)):
            newDF.iloc[i,j]=''
    display(newDF.iloc[1:,:-1])

def printNiceCorrTable(data, col1, col2):
        rho = data.corr().loc[col1, col2]
        pval = data.corr(method=lambda x, y: scipy.stats.pearsonr(x, y)[1]).loc[col1, col2]
        # display(np.round(pval,2))
        p = pval.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1] if x <= t]))
        # display(p)
        display(rho.round(2).astype(str) + p)