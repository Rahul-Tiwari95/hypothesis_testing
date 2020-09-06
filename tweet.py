# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:59:40 2020

@author: rahul
"""

import pandas as pd
import tweepy 

#Twitter API credentials
consumer_key = "jyX7VNW64XrZCJdoypi4weQ3P"
consumer_secret = "T9Pzd8lKsiGueJPEt8DRr3jbAU2aFu1NFSmxUqkjwm2YgYFdyB"
access_key = "567383948-yjxik45JBErEpDgO90AhqolGaZ0nabPb09OF0iTh"
access_secret = "ixDpe2heQoLCvU0fxVVruxEKJaTi1Ch5PAgDODfiptgZH"

alltweets = []	

def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))                
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    tweets_df = pd.DataFrame(columns = ["text"])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    
ShashiTharoor = get_all_tweets("ShashiTharoor")

