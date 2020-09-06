# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:41:06 2020

@author: rahul
"""

import pandas as pd
book = pd.read_csv("E:\\Data Science\\Data Sheet\\book rs.csv")
book.shape 
book.columns
book.rating

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words="english")

book["rating"].isnull().sum() 
book["rating"] = book["rating"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(anime.genre)   
tfidf_matrix.shape 

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

anime_index = pd.Series(anime.index,index=anime['name']).drop_duplicates()
anime_index["Hunter x Hunter (2011)"]

def get_anime_recommendations(Name,topN):
    
    anime_id = anime_index[Name]
    
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    anime_idx  =  [i[0] for i in cosine_scores_10]
    anime_scores =  [i[1] for i in cosine_scores_10]
    
    anime_similar_show = pd.DataFrame(columns=["name","Score"])
    anime_similar_show["name"] = anime.loc[anime_idx,"name"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.reset_index(inplace=True)  
    anime_similar_show.drop(["index"],axis=1,inplace=True)
    print (anime_similar_show)

    
get_anime_recommendations("Ginga Eiyuu Densetsu",topN=15)