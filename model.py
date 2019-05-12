import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import data_processing as dp
import turicreate as tc 

train_data = tc.SFrame(dp.ratings_train)
test_data = tc.SFrame(dp.ratings_test)

#Content based filtering
print "Content based filtering"
print "\n"
popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=[1],k=5)
popularity_recomm.print_rows(num_rows=25)


#Collaborative filtering
print "Collaborating filtering with cosine"
print "\n"
item_sim_model_cosine = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
item_sim_recomm = item_sim_model.recommend(users=[1],k=5)
item_sim_recomm.print_rows(num_rows=25)



print "Collaborating filtering with pearson"
print "\n"
item_sim_model_pearson = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')
item_sim_recomm = item_sim_model.recommend(users=[1],k=5)
item_sim_recomm.print_rows(num_rows=25)






