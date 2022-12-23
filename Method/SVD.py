#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O


# In[ ]:


# Reading ratings file
ratings = pd.read_csv('https://github.com/Bo-Yu-Columbia/6893_final_project_2022_16/tree/main/Method/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating', 'timestamp'])

# Reading users file
users = pd.read_csv('https://github.com/Bo-Yu-Columbia/6893_final_project_2022_16/blob/main/Method/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('https://github.com/Bo-Yu-Columbia/6893_final_project_2022_16/blob/main/Method/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])


# In[ ]:


movies.head()
ratings.head()

num_users = ratings.user_id.unique().shape[0]
num_movies = ratings.movie_id.unique().shape[0]
print 'Number of users = ' + str(num_users) + ' | Number of movies = ' + str(num_movies)


# In[ ]:


Ratings = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
Ratings.head()


# In[ ]:


R = Ratings.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)


# In[ ]:


#Model-Based Collaborative Filtering¶ -SVD

sparsity = round(1.0 - len(ratings) / float(num_users * num_movies), 3)
print 'The sparsity level of MovieLens1M dataset is ' +  str(sparsity * 100) + '%'


# In[ ]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Ratings_demeaned, k = 50)
sigma = np.diag(sigma)


# In[ ]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)
preds.head()


# In[ ]:


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 
    # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) 
    # This ID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movie_id',
               right_on = 'movie_id').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# In[ ]:


already_rated, predictions = recommend_movies(preds, 1, movies, ratings, 20)
# Top 20 movies that User 1 has rated 
already_rated.head(20)


# In[ ]:


# Top 20 movies that User 1310 hopefully will enjoy
predictions


# In[ ]:


# Model evaluation
# Import libraries from Surprise package
from surprise import Reader, Dataset, SVD, evaluate

# Load Reader library
reader = Reader()

# Load ratings dataset with Dataset library
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split the dataset for 5-fold evaluation
data.split(n_folds=5)


# In[ ]:


# Use the SVD algorithm.
svd = SVD()

# Compute the RMSE of the SVD algorithm.
evaluate(svd, data, measures=['RMSE'])


# In[ ]:


trainset = data.build_full_trainset()
svd.train(trainset)


# In[ ]:


ratings[ratings['user_id'] == 1]
svd.predict(1, 1600)

