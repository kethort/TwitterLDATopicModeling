#!/usr/bin/python
# https://dev.twitter.com/overview/api/response-codes
import sys
import tweepy
import unicodedata
import os
import ast
import json
import pyprind
from collections import defaultdict

# populate access credentials into list
def get_access_creds():
    credentials = []
    auths = []

    print('Building list of developer access credentials...')
    with open('twitter_dev_accounts.txt', 'r') as infile:
        for line in infile:
            if line.strip():
                credentials.append(line.strip())
            else:
                api = get_api(credentials)
                if(verify_working_credentials(api)):
                    auths.append(api)
    return auths

def get_api(credentials):
    consumer_key = credentials[0]
    consumer_secret = credentials[1]
    access_token = credentials[2]
    access_secret = credentials[3]
    
    auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    return api
    
def verify_working_credentials(api):
    verified = True

    try:
        api.verify_credentials()

    except tweepy.TweepError as e:
        verified = False

    except Exception as e:
        print(str(e))

    finally:
        return verified

def manage_auth_handlers(oauths):
    index = 0
    while True:
        api = oauths[index]

        try:
            limit = api.rate_limit_status()
            status_limit = limit['resources']['statuses']['/statuses/user_timeline']['remaining']
            if status_limit > 100:
                return api

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))

        finally:
            if index == (len(oauths) - 1):
                index = 0
            else:
                index += 1

def get_followers(user_id, api):
    followers = []
    while True:
        try:
            cursor = tweepy.Cursor(api.followers, id=user_id).pages()
            for page in cursor:
                followers += page

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))
                
        finally:
            return tweets

def unfollow_users(comm_set, api):
    ''' 
        requires developer read/write permissions
        enabled in application settings
    '''
    for user_id in comm_set:
        try:
            api.destroy_friendship(user_id) 
            print(str(developer_id) + ' unfollowed ' + str(user_id))

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))
    
def follow_users(comm_set, api):
    ''' 
        requires developer read/write permissions
        enabled in application settings
    '''
    for user_id in comm_set:
        try:
            api.create_friendship(user_id) 
            print(str(developer_id) + ' now following ' + str(user_id))

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))

def get_tweets(user_id, api):
    tweets = []
    while True:
        try:
            cursor = tweepy.Cursor(api.user_timeline, user_id).pages()
            for page in cursor:
                tweets += page

        except tweepy.TweepError as e:
            #print(e.message[0]['message'])
            pass

        except Exception as e:
            print(str(e))

        finally:
            return tweets
            
def user_status_count(user_id, api):
    count = 0

    try: 
        user = api.get_user(user_id=user_id)
        if(user.statuses_count):
            count = user.statuses_count

    except tweepy.TweepError as e:
        pass

    except Exception as e:
        print(str(e))

    finally:
        return count

def write_tweets(tweets, tweet_filename):
    with open(tweet_filename, 'w') as user_tweets:
        for tweet in tweets:
            user_tweets.write(tweet.text.encode("utf-8") + '\n')

def main(topology):
    inactive_users = {}
    active_users = {}
    oauths = get_access_creds()
    tweets_dir = './dnld_tweets/'

    with open(topology, 'r') as inp_file:
        comm_set = set(user for community in inp_file for user in ast.literal_eval(community))

    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)
    
    n = len(comm_set)
    bar = pyprind.ProgPercent(n, track_time=True, title='Downloading Tweets') 
    while comm_set:
        user = comm_set.pop()
        bar.update(item_id=user)
    
        api = manage_auth_handlers(oauths)

        # skip user who doesn't Tweet much 
        status_count = user_status_count(user, api)

        # skip user if you've already downloaded their tweets
        if os.path.exists(tweets_dir + str(user)):
            if status_count > 10:
                active_users[str(user)] = status_count
            else:
                inactive_users[str(user)] = status_count 
            continue

        tweets = get_tweets(user, api)

        if tweets:
            tweet_filename = tweets_dir + str(user)
            write_tweets(tweets, tweet_filename)

            if status_count > 10:
                active_users[str(user)] = status_count
            else:
                inactive_users[str(user)] = status_count 
        else:
                inactive_users[str(user)] = 0 

    with open('user_tweet_count.json', 'w') as outfile:
        json.dump(active_users, outfile, sort_keys=True, indent=4)

    with open('inactive_users.json', 'w') as outfile:
        json.dump(inactive_users, outfile, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
