# https://dev.twitter.com/overview/api/response-codes
import sys
import os
import tweepy
import unicodedata
import ast
import json
import pyprind
import numpy as np
import oauth_handler as auth
import matplotlib.pyplot as plt

''' a topology of twitter users is found based on follower relationships. networkx was used to
    find maximal cliques and discover communities derived from a clique based on set number of 
    "friends" to associate with. An example of what's in a topology file is in the img directory
    of project '''

def get_tweets(user_id, api):
    # try to get all of the tweets from the user's timeline
    # if there is an error move on, tweets will be empty and user will be added to inactive
    cursor = tweepy.Cursor(api.user_timeline, user_id).pages()
    while True:
        try:
            tweets = [page for page in cursor]

        except tweepy.TweepError as e:
            tweets = []
            api_codes = [401, 404, 500]
            if not str(e): break
            if(int(filter(str.isdigit, str(e))) in api_codes): break
            print('get_tweets: ' + str(e))

    return tweets

def user_status_count(user_id, api):
    try: 
        user = api.get_user(user_id=user_id)
        if(user.statuses_count):
            count = user.statuses_count

    except tweepy.TweepError as e:
        count = 0

    finally:
        return count

def write_tweets(tweets, tweet_filename):
    with open(tweet_filename, 'w') as user_tweets:
        for tweet in tweets:
            user_tweets.write(tweet.text.encode("utf-8") + '\n')

def read_json(file_name):
    try:
        with open(file_name, 'r') as comm_doc_vecs_file:
            return json.load(comm_doc_vecs_file)
    except:
        return {}

def write_json(tweets_dir, active_users, inactive_users):
    with open(os.path.join(tweets_dir, 'active_users.json'), 'w') as outfile:
        json.dump(active_users, outfile, sort_keys=True, indent=4)

    with open(os.path.join(tweets_dir, 'inactive_users.json'), 'w') as outfile:
        json.dump(inactive_users, outfile, sort_keys=True, indent=4)

def main(topology):
    # the input to main is the path to the topology file
    # the output to this script saves two json files inside the downloaded tweets directory,
    # one json file has all the active users the other has all inactive users from the topology
    # user activity is based on status count and availabilty of tweets (public vs private) 
    #
    # this script can be stopped and started in the middle of running it without losing progress

    inactive_users = read_json('dnld_tweets/inactive_users.json')
    active_users = read_json('dnld_tweets/active_users.json')
    oauths = auth.get_access_creds()
    tweets_dir = './dnld_tweets/'

    # put every single user (non repeating) from the topology file into a set
    with open(topology, 'r') as inp_file:
        comm_set = set(user for community in inp_file for user in ast.literal_eval(community))

    # create directory for storing tweets
    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)

    # download tweets for every single user in the set
    # separate active users from inactive users based on status count and availability
    bar = pyprind.ProgPercent(len(comm_set), track_time=True, title='Downloading Tweets') 
    while comm_set:
        user = comm_set.pop()
        bar.update(item_id=str(user) + '\t')

        if str(user) in inactive_users or str(user) in active_users:
            continue

        api = auth.manage_auth_handlers(oauths)

        # skip user if they don't exist or are inactive
        status_count = user_status_count(user, api)
        if status_count <= 10:
            inactive_users[str(user)] = status_count
            write_json(tweets_dir, active_users, inactive_users)
            continue

        # skip user if already downloaded their tweets
        if os.path.exists(os.path.join(tweets_dir, str(user))):
            active_users[str(user)] = status_count
            write_json(tweets_dir, active_users, inactive_users)
            continue

        tweets = get_tweets(user, api)

        if tweets:
            tweet_filename = tweets_dir + str(user)
            write_tweets(tweets, tweet_filename)
            active_users[str(user)] = status_count
        else:
            inactive_users[str(user)] = 0 

        write_json(tweets_dir, active_users, inactive_users)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
