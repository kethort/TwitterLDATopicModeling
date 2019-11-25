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
import pyprind
import logging
import argparse
import argcomplete

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
    bar = pyprind.ProgPercent(len(user_ids), track_time=True, title='Finding user followers') 
    for user in user_ids:
        bar.update(item_id=str(user) + '\t')
        api = auth.manage_auth_handlers(oauths)
        try: # protected tweets or user doesn't exist
            user_followers[user] = api.followers_ids(id=user)
        except:
            print("Skipping user: " + str(user))

    return user_followers

def save_networkx_graph(user_followers, filename):
    # create networkx graph from dictionary where the nodes are the keys
    # and the edges are the values <list>
    graph = nx.Graph()

    graph.add_nodes_from(user_followers.keys())

    for k, v in user_followers.items():
        graph.add_edges_from(([(k, t) for t in v]))
   
    # serialize the graph to disk
    data = json_graph.node_link_data(graph)

    with open(filename, 'w') as output:
        json.dump(data, output, sort_keys=True, indent=4)

def main():
    search_dir = 'twitter_geo_searches/'
    if not os.path.exists(os.path.dirname(search_dir)):
        os.makedirs(os.path.dirname(search_dir), 0o755)

    oauths = auth.get_access_creds()  

    # set up the command line arguments
    parser = argparse.ArgumentParser(description='Get twitter user ids and their follower ids from Tweepy and save in different formats')
    subparsers = parser.add_subparsers(dest='mode')
    
    search_parser = subparsers.add_parser('search', help='Gather Twitter user ids and followers by city, state and radius')
    search_parser.add_argument('-c', '--city', required=True, action='store', dest='city', help='City to search for Twitter user ids')
    search_parser.add_argument('-s', '--state', required=True, action='store', dest='state', help='State to search for Twitter user ids')   
    search_parser.add_argument('-r', '--radius', required=True, action='store', dest='radius', help='Radius to search Twitter API for user ids (miles or kilometers -- ex: 50mi or 50km)')   
    search_parser.add_argument('-f', '--filename', required=True, action='store', dest='filename', help='Name of output file for networkx graph data')   
    
    netwrkx_parser = subparsers.add_parser('netx', help='Perform operations on already generated networkx graph')
    netwrkx_parser.add_argument('-q', '--clique', action='store_true', dest='clique', help='Find cliques with networkx')
    netwrkx_parser.add_argument('-i', '--filename', required=True, action='store', dest='filename', help='Networkx input data filename')   

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == 'search':
        city = search_parser.city
        state = search_parser.state
        search_radius = search_parser.radius
        search_filename = search_parser.filename + '.json'

        # gets the first 50 zip codes by city and state
        zip_search = SearchEngine()
        zipcodes = zip_search.by_city_and_state(city, state, returns=50)

        user_ids = []
        user_followers = []
        # gets the user ids at each geo-location for the retrieved zip codes
        bar = pyprind.ProgPercent(len(zipcodes), track_time=True, title='Finding user ids') 
        for zipcode in zipcodes:
            bar.update(item_id=str(zipcode.zipcode) + '\t')
            latitude = zipcode.lat
            longitude = zipcode.lng
            user_ids.extend(get_user_ids(oauths, latitude, longitude, search_radius))
           
        # gets the followers of all the retrieved user ids 
        user_followers = get_user_followers(oauths, set(user_ids))
        
        save_networkx_graph(user_followers, os.path.join(search_dir, search_filename))

    if args.mode == 'netx':
        input_filename = netwrkx_parser.filename + '.json'

if __name__ == '__main__':
    sys.exit(main())
