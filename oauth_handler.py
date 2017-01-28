#!/usr/bin/python
# https://dev.twitter.com/overview/api/response-codes
import sys
import tweepy

def get_access_creds():
    '''
        Twitter API authentication credentials are stored in a file as:
        
            consumer_key
            consumer_secret
            access_token
            access_secret 
    '''
    credentials = []

    print('Building list of developer access credentials...')
    with open('twitter_dev_accounts.txt', 'r') as infile:
        for line in infile:
            if line.strip():
                credentials.append(line.strip())
            else:
                api = get_api(credentials)
                verify_working_credentials(api)
    return api

def get_api(credentials):
    consumer_key = credentials[0]
    consumer_secret = credentials[1]
    access_token = credentials[2]
    access_secret = credentials[3]
    
    auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

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
