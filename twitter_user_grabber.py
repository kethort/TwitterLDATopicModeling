import sys
import oauth_handler as auth

MAX_QUERIES = 100

def get_geolocation(oauths, location, scope):
    api = auth.manage_auth_handlers(oauths)
    places = api.geo_search(query=location, granularity=scope)
    
    location_match = places[0] # assuming that the first search result is best match
    
    latitude = location_match.centroid[1]
    longitude = location_match.centroid[0]
    return latitude, longitude

def get_user_ids(oauths, latitude, longitude, radius):
    tweets = []

    for i in range(0, MAX_QUERIES):
        api = auth.manage_auth_handlers(oauths)
        tweet_batch = api.search(q="*", rpp=1, geocode="%s,%s,%s" % (latitude, longitude, radius))
        tweets.extend(tweet_batch)
    
    return [tweet.author.id for tweet in tweets] 

def get_user_followers(oauths, user_ids):
    user_followers = {}
    for user in set(user_ids):
        api = auth.manage_auth_handlers(oauths)
        user_followers[user] = api.followers_ids(id=user)

    return user_followers

def main():
    oauths = auth.get_access_creds()

    latitude, longitude = get_geolocation(oauths, "Buford, GA", "city")
    print('latitude: %s longitude: %s' % (latitude, longitude)) 
    
    radius = "50mi" # mi or km

    user_ids = get_user_ids(oauths, latitude, longitude, radius)

    print(set(user_ids))
    
    user_followers = get_user_followers(oauths, user_ids) 

    print(user_followers)

if __name__ == '__main__':
    sys.exit(main())
