import sys
import os
import oauth_handler as auth
import json
import pprint
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import requests
import networkx as nx 
from networkx.readwrite import json_graph
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
import logging

''' Example script for getting twitter user topology by location '''

MAX_QUERIES = 100
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.CRITICAL)

def get_user_ids(oauths, latitude, longitude, radius):
    tweets = []

    for i in range(0, MAX_QUERIES):
        api = auth.manage_auth_handlers(oauths)
        try:
            tweet_batch = api.search(q="*", rpp=1, geocode="%s,%s,%s" % (latitude, longitude, radius))
            tweets.extend(tweet_batch)
        except Exception as e:
            print(e)

    return [tweet.author.id for tweet in tweets]

def get_user_followers(oauths, user_ids):
    # returns the followers of each user {user: [followers]}
    user_followers = {}
    for user in set(user_ids):
        api = auth.manage_auth_handlers(oauths)
        try: # protected tweets or user doesn't exist
            user_followers[user] = api.followers_ids(id=user)
        except:
            print("Skipping user: " + str(user))

    return user_followers

def followers_to_users_list(user_followers):
    user_ids = []
    for user in user_followers:
        user_ids += user_followers[user]
    return set(user_ids)

#def convert_followers_to_edges(followers, user_followers):
def get_nodes(user_followers):
    for node in user_followers:
        yield node

def get_edges(user_followers):
    for node in user_followers:
        for edge in user_followers[node]:
            yield (int(node), edge)

def build_network_graph(graph, nodes, edges):
    for node in nodes:
        graph.add_node(int(node))
    graph.add_edges_from(edges)

def main():
    search_dir = 'twitter_geo_searches/'
    if not os.path.exists(os.path.dirname(search_dir)):
        os.makedirs(os.path.dirname(search_dir), 0o755)

    search_filename = 'filename.json'
    
    oauths = auth.get_access_creds()
    location_api_key = 'Mx2ltANNNTll9Zk6OJRq4nOYIgDv4GDw9A46YfqKKs6nWmnDSPf1jNISTCGSvAjU'
    
    city = 'Newport Beach'
    state = 'CA'
    search_radius = "50mi" # mi or km

    zip_search = SearchEngine()
    zipcodes = zip_search.by_city_and_state(city, state, returns=50)

    geo_locations = []
    latitude = longitude = ''

    # make a tuple packed list of geo-locations from search result  
    for zipcode in zipcodes:
        latitude = zipcode.lat
        longitude = zipcode.lng
        geo_locations.append((latitude, longitude))

    user_ids = []
    user_followers = []

    bar = pyprind.ProgPercent(len(geo_locations), track_time=True, title='Finding user ids') 
    for location in geo_locations:
        latitude = location[0]
        longitude = location[1]
        print(len(user_ids)) 
        user_ids.extend(get_user_ids(oauths, latitude, longitude, search_radius))
        
    user_followers = get_user_followers(oauths, user_ids)
    
    with open(os.path.join(search_dir, search_filename), 'w') as outfile:
        json.dump(user_followers, outfile, sort_keys=True, indent=4)

    filename = os.path.join(search_dir, 'a.json')
    with open(filename, 'r') as twitter_users:
        user_followers = json.load(twitter_users)

    # create networkx graph from dictionary where the nodes are the keys
    # and the edges are the items in the list(value)
    graph = nx.Graph()

    graph.add_nodes_from(user_followers.keys())

    for k, v in user_followers.items():
        graph.add_edges_from(([(k, t) for t in v]))

    
    # serialize the graph to disk
    data = json_graph.node_link_data(graph)

    out_file = os.path.join(search_dir, 'graph_data.json')
    with open(out_file, 'w') as output:
        json.dump(data, output, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main())
