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
import argparse
import argcomplete

''' a topology of twitter users is found based on follower relationships. networkx was used to
    find maximal cliques and discover communities derived from a clique based on set number of
    "friends" to associate with. An example of what's in a topology file is in the img directory
    of project '''

def get_tweets(user_id, twpy_api):
    # try to get all of the tweets from the user's timeline
    # if there is an error move on, tweets will be empty and user will be added to inactive
    tweets = []

    try:
        for page in tweepy.Cursor(twpy_api.user_timeline, user_id).items():
            tweets.append(page.text)

    except tweepy.TweepError as e:
        pass

    return tweets

def user_status_count(user_id, twpy_api):
    try:
        user = twpy_api.get_user(user_id=user_id)
        if(user.statuses_count):
            count = user.statuses_count

    except tweepy.TweepError as e:
        count = 0

    finally:
        return count

def write_tweets(tweets, tweet_filename):
    with open(tweet_filename, 'w') as user_tweets:
        for tweet in tweets:
            if (int(sys.version.split('.')[0]) < 3): # python version less than 3
            	user_tweets.write(tweet.encode('utf-8') + '\n')
            else:
                user_tweets.write(tweet + '\n')

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

def main():
    # the output to this script saves two json files inside the downloaded tweets directory,
    # one json file has all the active users the other has all inactive users from the topology
    # user activity is based on status count and availabilty of tweets (public vs private)

    # script can be stopped and started in the middle of running it without losing progress
    parser = argparse.ArgumentParser(description='Get tweets of all twitter user ids in the provided topology file')
    parser.add_argument('-t', '--topology_file', required=True, action='store', dest='top_file', help='Location of topology file')
    parser.add_argument('-c', '--dev_creds', required=True, action='store', dest='dev_creds', help='Location of file containing Twitter developer access credentials')
    parser.add_argument('-o', '--output_dir', required=True, action='store', dest='output_dir', help='Name of the directory you want to download Tweets to')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    tweets_dir = args.output_dir

    # create directory for storing tweets
    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)

    inactive_users = read_json(os.path.join(tweets_dir, 'inactive_users.json'))
    active_users = read_json(os.path.join(tweets_dir, 'active_users.json'))
    twpy_api = auth.get_access_creds(args.dev_creds)

    if not twpy_api:
        print('Error: Twitter developer access credentials denied')
        return

    # put every single user (non repeating) from the topology file into a set
    with open(args.top_file, 'r') as inp_file:
        comm_set = set(user for community in inp_file for user in ast.literal_eval(community.replace('],', ']')))

    # download tweets for every single user in the set
    # separate active users from inactive users based on status count and availability
    bar = pyprind.ProgPercent(len(comm_set), track_time=True, title='Downloading Tweets')
    while comm_set:
        user = comm_set.pop()
        bar.update(item_id=str(user) + '\t')

        if str(user) in inactive_users or str(user) in active_users:
            continue

        # skip user if they don't exist or are inactive
        status_count = user_status_count(user, twpy_api)
        if status_count <= 10:
            inactive_users[str(user)] = status_count
            write_json(tweets_dir, active_users, inactive_users)
            continue

        # skip user if already downloaded their tweets
        if os.path.exists(os.path.join(tweets_dir, str(user))):
            active_users[str(user)] = status_count
            write_json(tweets_dir, active_users, inactive_users)
            continue

        tweets = get_tweets(user, twpy_api)

        if tweets:
            tweet_filename = os.path.join(tweets_dir, str(user))
            write_tweets(tweets, tweet_filename)
            active_users[str(user)] = status_count
        else:
            inactive_users[str(user)] = 0

        write_json(tweets_dir, active_users, inactive_users)

if __name__ == '__main__':
    sys.exit(main())
