import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import data_processing as dp
from sklearn.metrics.pairwise import pairwise_distances


def preprocessing():
    n_users = dp.ratings.user_id.unique().shape[0]
    print n_users
    n_items = dp.ratings.movie_id.unique().shape[0]
    print n_items

    data_matrix = np.zeros((n_users, n_items))
    print data_matrix.shape

    for line in dp.ratings.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]

    user_similarity = pairwise_distances(data_matrix, metric='cosine')
    print "user", user_similarity.shape
    item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
    print "item" , item_similarity.shape

    return data_matrix ,user_similarity , item_similarity


def prediction(type='user'):
    data_matrix , user_similarity , item_similarity = preprocessing()
    ratings = data_matrix
    if type =='user':
        mean_user_rating = ratings.mean(axis=1)
        print "mean", mean_user_rating.shape
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        print "rating diff", ratings_diff.shape 
        pred = mean_user_rating[:, np.newaxis] + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        print "user prediction" , pred.shape
    elif type == 'item':
        pred = ratings.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)]) 
        print "Item prediction" , pred.shape


preprocessing()
prediction('user')
prediction('item')
