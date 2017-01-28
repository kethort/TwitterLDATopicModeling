#!/usr/bin/python
import sys
import tweepy
import oauth_handler as auth

def unfollow_users(comm_set, api):
    ''' 
        requires developer read/write permissions
        enabled in Twitter dev application settings
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
        enabled in Twitter dev application settings
    '''
    for user_id in comm_set:
        try:
            api.create_friendship(user_id) 
            print(str(developer_id) + ' now following ' + str(user_id))

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))

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

def main(location):
    

if __name__ == '__main__':
    sys.exit(main(location))
