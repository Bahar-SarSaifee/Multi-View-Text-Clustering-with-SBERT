#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import re
import string
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('./Dataset/BBC News.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


Y = df.values[:, 2]


# In[ ]:


y_true = np.array(list(map(lambda x: 0 if x=="business" else 1 if x=="tech" else 2 if x=="politics" else 3 if x=="sport" else 4, Y)))


# In[ ]:


model1 = SentenceTransformer('all-MiniLM-L6-v2')

model2 = SentenceTransformer('all-distilroberta-v1')

model3 = SentenceTransformer('all-MiniLM-L12-v2')

model4 = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model5 = SentenceTransformer('paraphrase-albert-small-v2')

model6 = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# In[ ]:


def multi_view_clustering(documents, num_views, num_clusters):
    # Step 1: Vectorization for each view
    views = []
    for i in range(num_views):

        if i == 0:
            print(i)
            embeddings = model1.encode(documents)
        elif i == 1:
            print(i)
            embeddings = model2.encode(documents)
        elif i == 2:
            print(i)
            embeddings = model3.encode(documents)
        elif i == 3:
            print(i)
            embeddings = model4.encode(documents)
        elif i == 4:
            print(i)
            embeddings = model5.encode(documents)
        else:
            print(i)
            embeddings = model6.encode(documents)

        views.append(embeddings)
   
    # Step 2: Apply K-means for each view
    kmeans_models = []
    for view in views:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(view)
        kmeans_models.append(kmeans)

        
    # Step 3: Obtain cluster assignments for each document in each view
    cluster_assignments = []
    for i in range(num_views):
        cluster_assignments.append(kmeans_models[i].labels_)
        

    # Step 4: Apply aggregation technique (selective voting) to obtain the final clusters
    final_clusters = []
    num_documents = len(documents)
    for doc_index in range(num_documents):
        votes = np.zeros(num_clusters)
        for view_index in range(num_views):
            cluster_index = cluster_assignments[view_index][doc_index]
            votes[cluster_index] += 1
        
        # Apply selective voting
        max_vote = np.max(votes)
        max_clusters = np.where(votes == max_vote)[0]
        if len(max_clusters) == 1:
            # If there is a clear majority cluster, assign the document to that cluster
            final_cluster = max_clusters[0]
        else:
            # If there is a tie or no clear majority, assign the document to a random cluster among the top-voted clusters
            final_cluster = np.random.choice(max_clusters)
        
        final_clusters.append(final_cluster)

    return final_clusters


# In[ ]:


# Example usage

num_views = 6
num_clusters = 5

documents = df['Text']

clusters = multi_view_clustering(documents, num_views, num_clusters)
y_pred = clusters
print("Cluster assignments:", clusters[:10])


# In[ ]:


from sklearn.metrics import f1_score

print("F-measure:")
f1_score(y_true, y_pred, average='macro')


# In[ ]:


from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    confusion_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)

# Report Purity Score
purity = purity_score(y_true, y_pred)
print(f"The purity score is {round(purity*100, 2)}%")


# In[ ]:




