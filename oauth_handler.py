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

        with a space in between each set
    '''
    credentials = []
    oauths = []
    app_auths = []

    print('Building list of developer access credentials...')
    with open('twitter_dev_accounts.txt', 'r') as infile:
        for line in infile:
            if line.strip():
                credentials.append(line.strip())
            else:
                oauth_api, app_api = get_apis(credentials)
                if(verify_working_credentials(oauth_api)):
                    oauths.append(oauth_api)
                    app_auths.append(app_api)
                credentials = []
    return oauths, app_auths

def get_apis(credentials):
    consumer_key = credentials[0]
    consumer_secret = credentials[1]
    access_token = credentials[2]
    access_secret = credentials[3]
    
    auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    app_api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    #app_api = tweepy.API(auth)
    auth.set_access_token(access_token, access_secret)
    oauth_api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    #oauth_api = tweepy.API(auth)

    return oauth_api, app_api

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

def manage_auth_handlers(auths):
    index = 0
    while True:
        api = auths[index]

        try:
            limit = api.rate_limit_status()
            status_limit = limit['resources']['statuses']['/statuses/user_timeline']['remaining']
            if status_limit > 180:
                return api

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))

        finally:
            if index == (len(auths) - 1):
                index = 0
            else:
                index += 1

