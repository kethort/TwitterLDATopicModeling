#!/usr/bin/python
import sys
import pickle
import time
import tweepy
import unicodedata
import json
import os, os.path
import ast
import csv
import json
from collections import defaultdict
from re import split
from shutil import copyfile

index = 0

# populate access credentials into list
def get_access_creds():
    i = 0
    credentials = defaultdict(list)

    with open('twitter_dev_accounts.txt', 'r') as infile:
        for line in infile:
            if line.strip():
                credentials[i].append(line.strip())
            else:
                i += 1
    return credentials

# authenticates to the Twitter API and handles connection issues
def authenticate(credentials):
    global index
    print("Authentication in progress..." + str(index))

    # changes the access credentials each time the api rate limit has been exceeded
    while True:
        consumer_key = credentials[index][0]
        consumer_secret = credentials[index][1]
        access_token = credentials[index][2]
        access_secret = credentials[index][3]
                
        print(access_token)
        print(access_secret)
        print(consumer_key)
        print(consumer_secret)

        auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth)

        # handles connection status updates
        try:
            limit = api.rate_limit_status()
            status_limit = limit['resources']['statuses']['/statuses/user_timeline']['remaining']
            print(status_limit)
            if status_limit > 100:
                print("Authentication Completed\n")
                return api

        except Exception as e:
            print(str(e))

        finally:
            if index < (len(credentials) - 1):
                index += 1
            else:
                index = 0

# writes the text of all tweets to file
def write_tweets(tweets, tweet_filename):
    with open(tweet_filename, 'w') as user_tweets:
        for tweet in tweets:
            user_tweets.write(tweet.text.encode("utf-8") + '\n')

# writes the Tweet metadata being scraped to a file as:
# tweet_type, user_id, RT_user_id, RT_count, tweet_id, hashtags, screen_name
def write_tweet_meta(tweets, meta_filename, followers_filename):
    with open(meta_filename, 'a') as clique_tweet_metadata:
        for tweet in tweets:
            user_followers = {}
            favorite_count = tweet.favorite_count
            tweet_id = tweet.id_str
            screen_name = tweet.user.screen_name
            retweet_count = tweet.retweet_count
            user_id = tweet.user.id
            hashtags = []
            follower_count = tweet.user.followers_count
        
            # pickle dictionary to save memory
            if os.path.exists(followers_filename):
                with open(followers_filename, 'rb') as follower_dump:
                    user_followers = pickle.load(follower_dump)

            # get the follower count of each user
            if not any(str(user_id) in key for key in user_followers):
                user_followers[str(user_id)] = str(follower_count)
            
            # pickle dictionary to save memory
            with open(followers_filename, 'wb') as follower_dump:
                pickle.dump(user_followers, follower_dump)
                
            user_followers = {}

            # extract hashtags
            tagList = tweet.entities.get('hashtags')
            # check if there are hashtags
            if(len(tagList) > 0):
                for tag in tagList:
                    hashtags.append(tag['text'])
        
            # if the tweet is not a retweet
            if not hasattr(tweet, 'retweeted_status'):
                out = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('T', user_id, user_id, retweet_count, tweet_id, hashtags, screen_name) 
            # if it is retweet, get user id of original tweet 
            else:
                # must be defined in the else because if incoming tweet is not a retweet
                rt_user_id = tweet.retweeted_status.user.id
                rt_screen_name = tweet.retweeted_status.user.screen_name
                orig_tweet_id = tweet.retweeted_status.id_str
        
                out = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('RT', user_id, rt_user_id, retweet_count, orig_tweet_id, hashtags, rt_screen_name) 
            clique_tweet_metadata.write(out)

def get_followers(user_id, api, credentials):
    followers = []
    while True:
        try:
            cursor = tweepy.Cursor(api.followers, id=user_id, monitor_rate_limit=True, wait_on_rate_limit=True).pages()
            for page in cursor:
                followers += page

        except tweepy.TweepError as e:
            print(str(e))
                
        finally:
            return tweets, api

def get_tweets(user_id, api, credentials):
    tweets = []
    while True:
        try:
            cursor = tweepy.Cursor(api.user_timeline, user_id, monitor_rate_limit=True, wait_on_rate_limit=True).pages()
            for page in cursor:
                tweets += page

        except tweepy.TweepError as e:
            print(str(e))

        finally:
            return tweets, api

def build_comm_dirs(topology_file, tweets_dir):
    if not os.path.exists(os.path.dirname('tweets/')):
        os.makedirs(os.path.dirname('tweets/'))

    with open(topology_file, 'r') as infile:
        for i, community in enumerate(infile):
            comm_dir = 'tweets/community_' + str(i) + '/'
            if not os.path.exists(os.path.dirname(comm_dir)):
                os.makedirs(os.path.dirname(comm_dir))
            for user in ast.literal_eval(community):
                if os.path.exists(tweets_dir + str(user)):
                    copyfile(tweets_dir + str(user), comm_dir + str(user))
            

def get_comm_set(filename):
    comm_set = set()

    with open(filename, 'r') as inp_file:
        for community in inp_file:
            for user in ast.literal_eval(community):
                comm_set.add(user)

    return comm_set

def main(arg1):
    credentials = get_access_creds()

    tweets_dir = './dnld_tweets/'
    empty_dir = './empty_tweets/'

    comm_set = get_comm_set(str(arg1))

    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)
    
    if not os.path.exists(os.path.dirname(empty_dir)):
        os.makedirs(os.path.dirname(empty_dir), 0o755)

    while comm_set:
        user = comm_set.pop()
        print(user)
        
        if os.path.exists(tweets_dir + str(user)) or os.path.exists(tweets_dir + str(user)):
            continue

        api = authenticate(credentials)
        tweets,api = get_tweets(user, api, credentials)

        # don't do anything if tweet is empty
        if not tweets:
            tweet_filename = empty_dir + str(user)
            write_tweets(tweets, tweet_filename)
            continue

        tweet_filename = tweets_dir + str(user)
        write_tweets(tweets, tweet_filename)

    build_comm_dirs(arg1, tweets_dir)
                    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
