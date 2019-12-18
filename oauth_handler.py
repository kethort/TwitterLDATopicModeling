import sys
import tweepy
import pandas as pd

''' 
    This script opens a file with the user's twitter API authentication credentials and uses them to 
    gain access to the twitter API 
'''

def get_access_creds():
    #    Twitter API authentication credentials should be stored in a tab-separated (\t) file with:
    #        consumer_key \t consumer_secret \t access_token \t access_secret 

    credentials = pd.read_csv('twitter_dev_accounts', sep='\t', header=None, names=['consumer_key', 'consumer_secret', 'access_token', 'access_secret'])

    consumer_key = credentials['consumer_key'][0]
    consumer_secret = credentials['consumer_secret'][0]
    access_token = credentials['access_token'][0]
    access_secret = credentials['access_secret'][0]

    auth = tweepy.auth.OAuthHandler(str(consumer_key), str(consumer_secret))
    auth.set_access_token(str(access_token), str(access_secret))
    twpy_api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        
    if(verify_working_credentials(twpy_api)):
        print('Twitter developer keys verified')
        return twpy_api

def verify_working_credentials(api):
    verified = True
    try:
        api.verify_credentials()
    except tweepy.TweepError as e:
        verified = False
    finally:
        return verified

def main():
    get_access_creds()

if __name__ == '__main__':
    sys.exit(main())