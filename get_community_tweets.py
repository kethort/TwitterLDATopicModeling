#!/usr/bin/python
# https://dev.twitter.com/overview/api/response-codes
import sys
import pickle
import tweepy
import unicodedata
import os
import ast
import json
from collections import defaultdict

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

        except tweepy.TweepError as e:
            error = e.message[0]['code']
            # bad credentials
            if error == 32 or error == 89:
                #del credentials[index]
                continue
            elif error == 135:
                print('System time is incorrect. Cannot authenticate you.')
            elif error == 136:
                print('Your account has been blacklisted from using the API')
            else:
                print(str(e))

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
            # cursor = tweepy.Cursor(api.followers, id=user_id, monitor_rate_limit=True, wait_on_rate_limit=True).pages()
            cursor = tweepy.Cursor(api.followers, id=user_id).pages()
            for page in cursor:
                followers += page

        except tweepy.TweepError as e:
            error = e.message[0]['code']
            
            # user does not exist
            if error == 34:
                print('User does not exist')

            # rate limit reached
            elif error == 88 or error == 429:
                print('Rate limit reached. Reauthenticating.')
                api = authenticate(credentials)
                continue

            else:
                print(str(e))

        except Exception as e:
            print(str(e))
                
        finally:
            return tweets, api

def get_tweets(user_id, api, credentials):
    tweets = []
    while True:
        try:
            # cursor = tweepy.Cursor(api.user_timeline, user_id, monitor_rate_limit=True, wait_on_rate_limit=True).pages()
            cursor = tweepy.Cursor(api.user_timeline, user_id).pages()
            for page in cursor:
                tweets += page

        except tweepy.TweepError as e:
            error = e.message[0]['code']
            
            # user does not exist
            if error == 34:
                print('User does not exist')

            # rate limit reached
            elif error == 88 or error == 429:
                print('Rate limit reached. Reauthenticating.')
                api = authenticate(credentials)
                continue

            else:
                print(e)

        except Exception as e:
            print(e)

        finally:
            return tweets, api
            
def is_active_user(api, inactive_users, active_users, user_id):
    result = False
    
    try:
        user = api.get_user(user_id=user_id)
        if(user.statuses_count > 10):
            active_users[str(user_id)] = user.statuses_count
            result = True
        else:
            inactive_users[str(user_id)] = user.statuses_count

    except tweepy.TweepError, e:
        error = e.message[0]['code']

        # user not found or account suspended
        if error == 50 or error == 63 or error == 34:
            inactive_users[str(user_id)] = 0

        else:
            print(e)

    except Exception as e:
        print(e)

    finally:
        return result

def get_comm_set(filename):
    comm_set = set()

    with open(filename, 'r') as inp_file:
        for community in inp_file:
            for user in ast.literal_eval(community):
                comm_set.add(user)

    return comm_set

def main(arg1):
    inactive_users = {}
    active_users = {}
    credentials = get_access_creds()

    tweets_dir = './dnld_tweets/'

    comm_set = get_comm_set(str(arg1))

    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)
    
    while comm_set:
        user = comm_set.pop()
        print(user)
        
        api = authenticate(credentials)

        # don't waste time trying to download tweets for inactive user
        if not is_active_user(api, inactive_users, active_users, user):
            if not str(user) in inactive_users:
                inactive_users[str(user)] = 0 
            continue

        if os.path.exists(tweets_dir + str(user)):
            continue

        tweets,api = get_tweets(user, api, credentials)

        # tweepy exception handling not perfect, may still get empty tweets
        if tweets:
            tweet_filename = tweets_dir + str(user)
            write_tweets(tweets, tweet_filename)
        else:
            inactive_users[str(user)] = 0 

    with open('inactive_users.json', 'w') as outfile:
        json.dump(inactive_users, outfile, sort_keys=True, indent=4)
            
    with open('user_tweet_count.json', 'w') as outfile:
        json.dump(active_users, outfile, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
