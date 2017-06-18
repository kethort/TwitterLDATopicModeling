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

def get_tweets(user_id, api):
    tweets = []
    cursor = tweepy.Cursor(api.user_timeline, user_id).pages()

    while True:
        try:
            for page in cursor:
                tweets += page
            break

        except tweepy.TweepError as e:
            # 401 or 404 means they have restricted access on account
            if not str(e): break
            if(int(filter(str.isdigit, str(e))) == 401): break
            if(int(filter(str.isdigit, str(e))) == 404): break
            print('get_tweets: ' + str(e))
            pass

    return tweets
            
def user_status_count(user_id, api):
    count = 0
    try: 
        user = api.get_user(user_id=user_id)
        if(user.statuses_count):
            count = user.statuses_count

    except tweepy.TweepError as e:
        #print(e.message[0]['message'])
        pass

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
    inactive_users = read_json('dnld_tweets/inactive_users.json')
    active_users = read_json('dnld_tweets/active_users.json')
    oauths = auth.get_access_creds()
    tweets_dir = './dnld_tweets/'

    with open(topology, 'r') as inp_file:
        comm_set = set(user for community in inp_file for user in ast.literal_eval(community))

    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)
    
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

        # skip user if you've already downloaded their tweets
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
#    user_tweet_distribution()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
