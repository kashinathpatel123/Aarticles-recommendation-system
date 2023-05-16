#!/usr/bin/env python
# coding: utf-8

# Article Recommendation System
# There are many ways to create recommendation systems. To create an articles recommendation system, we need to focus on content rather than user interest. For example, if a user reads an article based on clustering, all recommended articles should also be based on clustering. So to recommend articles based on the content:
# 
# we need to understand the content of the article
# match the content with all the other articles
# and recommend the most suitable articles for the article that the reader is already reading
# For this task, we can use the concept of cosine similarity in machine learning. Cosine similarity is a method of building recommendation systems based on the content. It is used to find similarities between two different pieces of text documents. So we can use cosine similarity to build an article recommendation system. In the section below, I will take you through how to build an article recommendation system with machine learning using Python.

# Article Recommendation System using Python
# To create an article recommendation system, I collected data about some of the articles on this website itself. So let’s import the necessary Python libraries and the dataset we need to create an articles recommendation system:

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/articles.csv", encoding='latin1')
data.head()


# This dataset is completely ready to use to create a recommender system, so let’s use the cosine similarity algorithm and write a Python function to recommend articles:

# In[3]:


articles = data["Article"].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
def recommend_articles(x):
    return ", ".join(data["Title"].loc[x.argsort()[-5:-1]])    
data["Recommended Articles"] = [recommend_articles(x) for x in uni_sim]
data.head()


# As you can see from the output above, a new column has been added to the dataset that contains the titles of all the recommended articles. Now let’s see all the recommendations for an article:

# In[4]:


print(data["Recommended Articles"][22])


# Index 22 contains an article on “agglomerated clustering”, and all the recommended articles are also based on the concepts of clustering, so we can say that this recommender system can also give great results in real-time.

# In[ ]:


https://thecleverprogrammer.com/2021/11/10/article-recommendation-system-with-machine-learning/

