import sys
import os
import oauth_handler as auth
import json
import networkx as nx
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

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

#def convert_followers_to_edges(followers, user_followers):
def get_nodes(user_followers):
    for node in user_followers:
        yield node

def get_edges(user_followers):
    for node in user_followers:
        for edge in user_followers[node]:
            yield (int(node), edge)
    
#def dict_to_iter(user_followers):
#    for user in user_followers:
#        yield user_followers[user]

def build_network_graph(graph, nodes, edges):
    for node in nodes:
        graph.add_node(int(node))
    graph.add_edges_from(edges)

def main():
    '''
    oauths = auth.get_access_creds()

    latitude, longitude = get_geolocation(oauths, "Buford, GA", "city")

    radius = "50mi" # mi or km

    user_ids = get_user_ids(oauths, latitude, longitude, radius)

    user_followers = get_user_followers(oauths, user_ids) 

    search_dir = 'twitter_geo_searches/'
    if not os.path.exists(os.path.dirname(search_dir)):
        os.makedirs(os.path.dirname(search_dir), 0o755)

    filename = str(latitude) + '_' + str(longitude) + '.json'
    
    with open(os.path.join(search_dir, filename), 'w') as outfile:
        json.dump(user_followers, outfile, sort_keys=True, indent=4)
    
    with open(os.path.join(search_dir, filename), 'r') as twitter_users:
    '''
    search_dir = 'twitter_geo_searches/'
    with open(os.path.join(search_dir, '34.1180826_-83.9969963411.json'), 'r') as twitter_users:
        user_followers = json.load(twitter_users)

    nodes = [node for node in get_nodes(user_followers)]
    edges = [edge for edge in get_edges(user_followers)]

    #pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    #func = partial(convert_followers_to_edges, dict_to_iter(user_followers))
    #edges = pool.imap(func, user_followers)
    #pool.close()
    #pool.join()

    #print(edges[0])

    graph = nx.Graph()
    build_network_graph(graph, nodes, edges)

    for cliques in nx.find_cliques(graph):
        print(cliques)
    #plt.figure(figsize=(18, 18))
    #plt.axis('off')

    #nx.draw_networkx(graph)

    
if __name__ == '__main__':
    sys.exit(main())
