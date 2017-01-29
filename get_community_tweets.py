#!/usr/bin/python
# https://dev.twitter.com/overview/api/response-codes
import sys
import os
import tweepy
import unicodedata
import ast
import json
import pyprind
import oauth_handler as auth

def get_tweets(user_id, api):
    tweets = []
    cursor = tweepy.Cursor(api.user_timeline, user_id).pages()

    while True:
        try:
            for page in cursor:
                tweets += page
            break

        except tweepy.TweepError as e:
            #print(e.message[0]['message'])
            continue

        except Exception as e:
            print(str(e))

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
    _, app_auths = auth.get_access_creds()
    tweets_dir = './dnld_tweets/'

    with open(topology, 'r') as inp_file:
        comm_set = set(user for community in inp_file for user in ast.literal_eval(community))

    if not os.path.exists(os.path.dirname(tweets_dir)):
        os.makedirs(os.path.dirname(tweets_dir), 0o755)
    
    bar = pyprind.ProgPercent(len(comm_set), track_time=True, title='Downloading Tweets') 
    while comm_set:
        user = comm_set.pop()
        bar.update(item_id=user)
    
        api = auth.manage_auth_handlers(app_auths)

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
