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
from networkx.algorithms.community import k_clique_communities
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
import pyprind
import logging
import argparse
import argcomplete

''' Script for getting twitter user topology by location '''

MAX_QUERIES = 200
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.CRITICAL)

def build_netx_graph(user_followers):
	# generates a networkx graph from a dictionary containing users and their followers 
    print('Building networkx graph from user followers dictionary')
    graph = nx.Graph()
    graph.add_nodes_from(user_followers.keys())

    for k, v in user_followers.items():
        graph.add_edges_from(([(int(k), t) for t in v]))

    return graph

def generate_cliques(graph, filename, min_size=4):
    # generates a topology of maximal cliques from a given networkx graph
    # this process eats up lots of memory and takes a long to complete on very large datasets
    print('Generating cliques from networkx graph')
    data = []

    for clique in nx.find_cliques(graph):
    	if len(list(clique)) > min_size:
    		data.extend([list(clique)])

    with open(filename, 'w') as output:
        for clique in data:
            output.write(str(clique) + '\n')


def generate_communities(graph, filename, min_size=6):
	# generates a topology of communities from a given networkx graph
    print('Generating communities from networkx graph')
    data = []

    for community in k_clique_communities(graph, min_size):
    	data.extend([list(community)])

    with open(filename, 'w') as output:
    	for community in data:
            output.write(str(community) + '\n')

def open_nx_graph(filename):
    data = {}
    with open(filename, 'r') as graph_data:
        data = json.load(graph_data)

    return json_graph.node_link_graph(data)

def pythonify_dict(data):
    # converts all keys in a dictionary to integers
    for k in data:
        value = data[k]
        try:
            newkey = int(k)
            del data[k]
            k = newkey
        except Exception as e:
            print(str(e))
            pass

        data[k] = value

def read_json(filename):
    try:
        with open(filename, 'r') as infile:
            return json.load(infile)
    except:
        return []

def write_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile + '.json', sort_keys=True)

def get_directory_of_file(filename):
    filename_loc = len(filename.strip('/').split('/')) - 1
    working_dir = filename.strip('/').split('/')[0:filename_loc]
    working_dir = '/'.join(working_dir) + '/'

    return working_dir

def get_user_ids(twpy_api, latitude, longitude, radius):
    tweets = []

    for i in range(0, MAX_QUERIES):
        try:
            tweet_batch = twpy_api.search(q="*", rpp=1, geocode="%s,%s,%s" % (latitude, longitude, radius))
            tweets.extend(tweet_batch)
        except Exception as e:
            print(e)

    return [tweet.author.id for tweet in tweets]

def get_user_followers(twpy_api, working_dir, filename, user_ids):
    # returns the followers of each user {user: [followers]} and also updates/returns user ids
    user_followers = {}
    bar = pyprind.ProgPercent(len(user_ids), track_time=True, title='Finding user followers')
    for user in user_ids:
        bar.update(item_id=str(user) + '\t')
        try: # protected tweets or user doesn't exist
            user_followers[user] = twpy_api.followers_ids(id=user)
        except Exception as e:
            print("Skipping user: " + str(user))
        else:
            current_followers = read_json(os.path.join(working_dir, filename + '.json'))
            # this conditional handles the first time that the file is written to
            if isinstance(current_followers, dict):
                current_followers[str(user)] = user_followers[user]
                write_json(os.path.join(working_dir, filename), current_followers)
            else:
                write_json(os.path.join(working_dir, filename), user_followers)

    return user_followers

def collect_user_followers(depth, twpy_api, working_dir, filename, user_ids):
    # the depth value is how far into the user follower relationship to collect user followers
    # if n == 2 then the list of users will grow by two times the depth of the amount of followers collected 
    # and the entire collection of those users followers will compose the resulting dictionary
    for i in range(0, int(depth)):
        user_followers = get_user_followers(twpy_api, working_dir, filename, set(user_ids))

        # update the list of user_ids to include the followers 
        for user in user_followers:
            user_ids.extend(user_followers[user])

        write_json(filename + '-users', list(set(user_ids)))
        write_json(os.path.join(working_dir, filename), user_followers)

def convert_followers_to_users(input_file, out_file, working_dir):
    # flattens a dictionary of {user: [followers]} into a single list of users
    print('Converting user followers to users')
    user_followers = read_json(input_file)
    users = []

    for k, v in user_followers.items():
        users.extend(k)
        users.extend(t for t in v)

    users = list(set(users))

    save_path = os.path.join(working_dir, out_file)
    write_json(save_path, users)

def main():
    # set up the command line arguments
    parser = argparse.ArgumentParser(description='Get twitter user ids and their follower ids using Tweepy and save in different formats')
    subparsers = parser.add_subparsers(dest='mode')

    search_parser = subparsers.add_parser('search', help='Gather Twitter user ids by city, state and radius')
    search_parser.add_argument('-c', '--city', required=True, action='store', dest='city', help='City to search for Twitter user ids. REQUIRED')
    search_parser.add_argument('-s', '--state', required=True, action='store', dest='state', help='State to search for Twitter user ids. REQUIRED')
    search_parser.add_argument('-r', '--radius', required=True, action='store', dest='radius', help='Radius to search Twitter API for user ids (miles or kilometers -- ex: 50mi or 50km). REQUIRED')
    search_parser.add_argument('-d', '--depth', required=True, action='store', dest='depth', help='This value represents how far to traverse into user follower relationships when gathering users. REQUIRED')
    search_parser.add_argument('-f', '--filename', required=True, action='store', dest='filename', help='Name of output file to store gathered users in. REQUIRED')
    search_parser.add_argument('-z', '--creds', required=True, action='store', dest='creds', help='Path to Twitter developer access credentials REQUIRED')

    continue_parser = subparsers.add_parser('getfws', help='Takes in already gathered jsonified list of users and retrieves their followers')
    continue_parser.add_argument('-f', '--filename', action='store', help='Filename of the previously saved Twitter users ids in .json format')
    continue_parser.add_argument('-d', '--depth', required=True, action='store', dest='depth', help='This value represents how far to traverse into user follower relationships when searching for followers. REQUIRED')
    continue_parser.add_argument('-z', '--creds', required=True, action='store', dest='creds', help='Path to Twitter developer access credentials REQUIRED')

    convert_parser = subparsers.add_parser('convert', help='Convert user followers dict to users list and save file. This is the file format used when continuing the get followers function and in get_community_tweets.py')
    convert_parser.add_argument('-i', '--input_file', action='store', help='Filename of the previously saved followers dictionary')
    convert_parser.add_argument('-o', '--out_file', action='store', help='Filename to store the output. Just the filename no path is needed. The output file will be saved in the folder of the input file')

    netx_parser = subparsers.add_parser('netx', help='Create cliques or communities from user follower data')
    group = netx_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-q', '--gen_cliques', required=False, action='store_true', dest='gen_cliques', help='Generate cliques from user followers dictionary')
    group.add_argument('-c', '--gen_comms', required=False, action='store_true', dest='gen_comms', help='Generate communities from user followers dictionary')
    netx_parser.add_argument('-n', '--min_size', action='store', dest='min_size', nargs='?', type=int, const=1, default=4, help='Constraint for min size of clique or community (default is 4)')
    netx_parser.add_argument('-i', '--in_filename', required=True, action='store', dest='in_filename', help='User followers dictionary file REQUIRED')
    netx_parser.add_argument('-o', '--out_filename', required=True, action='store', dest='out_filename', help='Output topology filename REQUIRED')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == 'convert':
        working_dir = get_directory_of_file(args.input_file)
        convert_followers_to_users(args.input_file, args.out_file, working_dir)

    if args.mode == 'getfws':
        twpy_api = auth.get_access_creds(args.creds)

        if not twpy_api:
            print('Error: Twitter developer access credentials denied')
            return

        working_dir = get_directory_of_file(args.filename)

        user_ids = read_json(args.filename)
        if not user_ids:
        	print('Error: No users found in provided file')
        	return

        # gets the followers of all the retrieved user ids 'depth' number of times
        collect_user_followers(args.depth, twpy_api, working_dir, args.filename, user_ids)

    if args.mode == 'search':
        twpy_api = auth.get_access_creds(args.creds)

        if not twpy_api:
            print('Error: Twitter developer access credentials denied')
            return

        working_dir = get_directory_of_file(args.filename)

        # gets the first 50 zip codes by city and state
        zip_search = SearchEngine()
        zipcodes = zip_search.by_city_and_state(args.city, args.state, returns=50)

        user_ids = []
        user_followers = []
        # gets the user ids at each geo-location for the retrieved zip codes
        bar = pyprind.ProgPercent(len(zipcodes), track_time=True, title='Finding user ids')
        for zipcode in zipcodes:
            bar.update(item_id='zip code:' + str(zipcode.zipcode) + '\t')
            user_ids.extend(get_user_ids(twpy_api, zipcode.lat, zipcode.lng, args.radius))
            write_json(args.filename, list(set(user_ids)))

    if args.mode == 'netx':
        user_followers = read_json(args.in_filename)
        pythonify_dict(user_followers)
        print("Number of followers: " + str(len(user_followers)))
        output_filename = args.out_filename + '.json'
        graph = build_netx_graph(user_followers)

        if args.gen_cliques:
            generate_cliques(graph, output_filename, args.min_size)
        if args.gen_comms:
            generate_communities(graph, output_filename, args.min_size)

if __name__ == '__main__':
    sys.exit(main())
