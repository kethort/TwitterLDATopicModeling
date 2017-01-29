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

def get_followers(user_id, app_auths):
    followers = []
    api = auth.manage_auth_handlers(app_auths)
    
    cursor = tweepy.Cursor(api.followers_ids, id=user_id).pages()
    print(user_id)
    while True:
        try:
            for page in cursor:
                followers += page
            break

        except tweepy.TweepError as e:
            continue

        except StopIteration:
            break
                
    print(len(followers))
    return followers

def get_user_ids(search_query, location, app_auths):
    users = [] 
    api = auth.manage_auth_handlers(app_auths)

    cursor = tweepy.Cursor(api.search, q=search_query).items()
    while True:
        try:
            for item in cursor:
                user = item.user.id
                print(user)
                if user not in users:
                    users.append(user)
            if len(users) > 10000:
                break
            
        except tweepy.TweepError as e:
            continue

        except StopIteration:
            break

    print('length: ' + str(len(users)))     
    return users

def main(location):
    users_by_loc = {}
    oauths, app_auths = auth.get_access_creds()

    api = auth.manage_auth_handlers(oauths)
    places = api.geo_search(query=location)
    search_query = 'place:' + str(places[0].id)

    users = get_user_ids(search_query, location, app_auths)

    for user in users:
        if user not in users_by_loc:
            users_by_loc[user] = get_followers(user, app_auths)

    with open('users_in_' + location, 'w') as outfile:
        json.dump(users_by_loc, outfile, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
