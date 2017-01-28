#!/usr/bin/python
import sys
import tweepy
import json
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
            cursor = tweepy.Cursor(api.followers_ids, id=user_id).pages()
            for page in cursor:
                followers.extend(page)

        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))
                
        finally:
            return followers

def get_user_ids(location, num_tweets, api):
    users = [] 
    places = api.geo_search(query=location)
    search_query = 'place:' + str(places[0].id)

    while True:
        try:
            items = tweepy.Cursor(api.search, q=search_query).items(num_tweets)
            for item in items:
                user = item.user.id
                print(user)
                if user not in users:
                    users.append(item.user.id)
            
        except tweepy.TweepError as e:
            pass

        except Exception as e:
            print(str(e))
                
        finally:
            return users

def main(location, num_users):
    api = auth.get_access_creds()

    users_by_loc = {}
    users = get_user_ids(location, num_users, api)

    for user in users:
        if user not in users_by_loc:
            api = auth.manage_auth_handlers(oauths)
            users_by_loc[user] = get_followers(user, api)

    with open('users_in_' + location, 'w') as outfile:
        json.dump(users_by_loc, outfile, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
