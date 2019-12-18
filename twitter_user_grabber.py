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
import ast
from networkx.algorithms import community
from networkx.readwrite import json_graph
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
import pyprind
import logging
import argparse
import argcomplete

''' Example script for getting twitter user topology by location '''

MAX_QUERIES = 100
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.CRITICAL)

def get_user_ids(twpy_api, latitude, longitude, radius):
    tweets = []

    for i in range(0, MAX_QUERIES):
        try:
            tweet_batch = twpy_api.search(q="*", rpp=1, geocode="%s,%s,%s" % (latitude, longitude, radius))
            tweets.extend(tweet_batch)
        except Exception as e:
            print(e)

    return [tweet.author.id for tweet in tweets]

def get_user_followers(twpy_api, user_ids):
    # returns the followers of each user {user: [followers]} and also updates/returns user ids
    followers = user_ids
    user_followers = {}
    bar = pyprind.ProgPercent(len(user_ids), track_time=True, title='Finding user followers') 
    for user in user_ids:
        bar.update(item_id=str(user) + '\t')
        try: # protected tweets or user doesn't exist
            user_followers[user] = twpy_api.followers_ids(id=user)
            followers.extend(user_followers[user])
        except:
            print("Skipping user: " + str(user))

    return set(followers), user_followers

def save_user_follower_networkx_graph(user_followers, filename):
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

def gather_cliques(clique):
    return clique

def open_nx_graph(filename):
    data = {}
    with open(filename, 'r') as graph_data:
        data = json.load(graph_data)

    return json_graph.node_link_graph(data)

def main():
    search_dir = 'twitter_geo_searches/'
    if not os.path.exists(os.path.dirname(search_dir)):
        os.makedirs(os.path.dirname(search_dir), 0o755)

    twpy_api = auth.get_access_creds()  
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))

    # set up the command line arguments
    parser = argparse.ArgumentParser(description='Get twitter user ids and their follower ids from Tweepy and save in different formats')
    subparsers = parser.add_subparsers(dest='mode')
    
    search_parser = subparsers.add_parser('search', help='Gather Twitter user ids and followers by city, state and radius')
    search_parser.add_argument('-c', '--city', required=True, action='store', dest='city', help='City to search for Twitter user ids. REQUIRED')
    search_parser.add_argument('-s', '--state', required=True, action='store', dest='state', help='State to search for Twitter user ids. REQUIRED')   
    search_parser.add_argument('-r', '--radius', required=True, action='store', dest='radius', help='Radius to search Twitter API for user ids (miles or kilometers -- ex: 50mi or 50km). REQUIRED')   
    search_parser.add_argument('-f', '--filename', required=True, action='store', dest='filename', help='Name of output file for networkx graph data. REQUIRED')   
    
    netx_parser = subparsers.add_parser('netx', help='Perform operations on already generated networkx graph')
    netx_parser.add_argument('-q', '--clique', action='store_true', help='Find cliques with networkx')
    netx_parser.add_argument('-x', '--clq_filename', action='store', help='Provide a filename for the serialized output of find_cliques')
    netx_parser.add_argument('-g', '--graph_filename', required=True, action='store', dest='graph_filename', help='Networkx input data filename. REQUIRED')   
    netx_parser.add_argument('-o', '--out_filename', required=True, action='store', dest='out_filename', help='Networkx output data filename REQUIRED')      
    netx_parser.add_argument('-k', '--comm', action='store_true', help='Find communities with networkx')
    netx_parser.add_argument('-p', '--print_graph', action='store_true', help='Print networkx graph')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    if not args.mode:
        print('ERROR: No arguments provided. Use -h or --help for help')
        return

    if args.mode == 'search':
        city = args.city
        state = args.state
        search_radius = args.radius
        search_filename = args.filename + '.json'

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
            user_ids.extend(get_user_ids(twpy_api, latitude, longitude, search_radius))
           
        n = 2
        # gets the followers of all the retrieved user ids n number of depths
        for i in range(0, n):
            user_ids, user_followers = get_user_followers(twpy_api, set(user_ids))
        
        filename = os.path.join(search_dir, search_filename)
        save_user_follower_networkx_graph(user_followers, filename)

    if args.mode == 'netx':
        graph_filename = os.path.join(search_dir, args.graph_filename + '.json')
        output_filename = os.path.join(search_dir, args.out_filename + '.json')
        graph = open_nx_graph(graph_filename)
        cliques = []

        if args.clique: 
            for clique in pool.map(gather_cliques, nx.find_cliques(graph)):
                cliques.append([int(member) for member in clique])

            with open(output_filename, 'w') as output:
                for clique in cliques:
                    output.write('%s,\n' % (clique))

        elif args.comm:
            if args.clq_filename:
                clique_filename = os.path.join(search_dir, args.clq_filename + '.json')
                # load the clique topology file
                with open(clique_filename, 'r') as find_cliques_file:
                    cliques = [clique for cliques in find_cliques_file for clique in ast.literal_eval(cliques)]

            with open(output_filename, "w") as output:  
                for node in pool.map(gather_cliques, community.girvan_newman(graph)):
                    print(node)
                    #output.write(str([int(item) for item in node]) + ', \n')      
        elif args.print_graph: 
            nx.draw(graph)
            plt.show()     


    print("Job complete")

if __name__ == '__main__':
    sys.exit(main())
