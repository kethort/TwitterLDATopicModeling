#!/usr/bin/python
# https://dev.twitter.com/overview/api/response-codes
import sys, os
import tweepy

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
                credentials = []
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

