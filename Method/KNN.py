#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


# In[ ]:


user_url=r"https://drive.google.com/file/d/155oakHcdg3UWi9I-mKI4FWNGZM0G1knS/view?usp=share_link"
mov_url=r"https://drive.google.com/file/d/1jFVDbi43OkinguO6URYlRf41nTbhRx6W/view?usp=share_link"

movcol=['movieId','title']
df_movies = pd.read_csv(mov_url,names=movcol,sep='|',encoding='latin-1', usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
df_movies.head()


# In[ ]:


usercol=['userId','movieId','rating']
df_ratings = pd.read_csv(user_url,names=usercol,sep='\t',encoding='latin-1', usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
df_ratings.head

movies_users=df_ratings.pivot(index='movieId', columns='userId',values='rating').fillna(0)
mat_movies_users=csr_matrix(movies_users.values)
model_knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)

model_knn.fit(mat_movies_users)

def recommender(movie_name, data, model, n_recommendations ):
    model.fit(data)
    idx=process.extractOne(movie_name, df_movies['title'])[2]
    print('Movie Selected: ',df_movies['title'][idx], 'Index: ',idx)
    print('Searching for recommendations.....')
    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
    for i in indices:
        print(df_movies['title'][i].where(i!=idx))
    
recommender('Vertigo', mat_movies_users, model_knn,20)

