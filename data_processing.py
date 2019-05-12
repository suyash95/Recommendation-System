import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

#user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('dataset/ml-100k/u.user', sep ='|' , names=u_cols , encoding='latin-1')
print(users.shape)
print(users.head())

#rating file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('dataset/ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
print(ratings.shape)
print(ratings.head())

#item file
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('dataset/ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')
print(items.shape)
print(items.head())

ratings_train = pd.read_csv('dataset/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('dataset/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
#print ratings_train.shape, ratings_test.shape