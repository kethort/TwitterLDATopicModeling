import ast
import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import argcomplete
import matplotlib.pyplot as plt
import plot_distances as pltd
import tweets_on_LDA as tlda
import multiprocessing
from functools import partial
from gensim import utils, corpora, models

''' This script aggregates all the tweets of each entire community and clique into a single document and then 
    generates the topic probability distribution using that document against a given LDA model. 
    It then plots each communities topic probability distribution as well as plots comparisons between the JSD of individual users
    against a community vs individual users against a clique. These graphs are useful for determining the relationship of familiarity 
    between larger vs smaller social groups and their members. 

    NOTE: assumes you have downloaded tweets from a topology of cliques and the communities that are created from the cliques using
    networkx. the steps outlined on description page of github should be complete for both clique and community topologies
'''

def aggregate_tweets(i, clique, tweets_dir, output_dir):
    if not os.path.exists(output_dir + 'clique_' + str(i)):
        print('Aggregating tweets for clique_' + str(i))
        with open(output_dir + 'clique_' + str(i), 'w') as outfile:
            for user in ast.literal_eval(clique):
                if os.path.exists(tweets_dir + str(user)):
                    with open(tweets_dir + str(user)) as tweet:
                        for line in tweet:
                            outfile.write(str(line))

def draw_dist_graph(clique_name, **kwargs):
    if not os.path.exists(kwargs['output_dir'] + clique_name + '.png'):
        try:
            doc_vector = kwargs['doc_vecs'][clique_name]
        except KeyError:
            return

        print('Drawing probability distribution graph for ' + clique_name)
        x_axis = [topic_id + 1 for topic_id, dist in enumerate(doc_vector)]
        y_axis = [dist for topic_id, dist in enumerate(doc_vector)]
        plt.bar(x_axis, y_axis, width=1, align='center', color='r')
        plt.xlabel('Topics')
        plt.ylabel('Probability')
        plt.title('Topic Distribution for clique')
        plt.xticks(np.arange(2, len(x_axis), 2), rotation='vertical', fontsize=7)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.max(y_axis) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(kwargs['output_dir'] + clique_name)
        plt.close()

def draw_user_to_clique_graphs(distance_dir, dist_file):
    if not os.path.exists(distance_dir + dist_file + '.png'):
        print('Drawing community members distance from clique for: ' + dist_file)
        df = pd.read_csv(distance_dir + dist_file, sep='\t', header=None, names=['user', 'clique', 'distance'])
        x_axis = [str(row['user']) for idx, row in df.iterrows()]
        y_axis = [float(row['distance']) for idx, row in df.iterrows()]
        plt.figure(figsize=(20, 10))
        plt.bar(np.arange(1, len(x_axis) + 1, 1), y_axis, width=1, align='center', color='r')
        plt.xlabel('Community Users')
        plt.ylabel('Divergence from Clique')
        plt.title('Distances from ' + dist_file + ' to the clique where grown from', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(0, len(x_axis) + 1, 2), np.arange(0, len(x_axis) + 1, 2), rotation='vertical', fontsize=8)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.log(2) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(distance_dir + dist_file)
        plt.close()

def draw_community_median_distances(user_topics_dir, distance_file, df):
    y_axis = [row['avg_distance'] for idx, row in df.iterrows()]
    x_axis = np.arange(0, len(y_axis))
    plt.figure(figsize=(20, 10))
    plt.plot(x_axis, y_axis, 'r')
    plt.fill_between(x_axis, y_axis, color='red', alpha=0.5)
    plt.xlabel('Community ID')
    plt.ylabel('Divergence from Clique')
    plt.title('Community Users Divergence from Clique', fontsize=14, fontweight='bold')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.subplots_adjust(bottom=0.2)
    plt.ylim([0, np.log(2) + 0.01])
    plt.xlim([0, len(x_axis) - 1])
    plt.savefig(distance_file)
    plt.close()

def cliques_to_iter(tweet_folder):
    for path, dirs, files in os.walk(tweet_folder):
        files.sort()
        for filename in files:   
            yield filename
        break 

def distance_files_to_iter(distance_dir):
    for dist_file in os.listdir(distance_dir):
        if not dist_file.endswith('.png'):
            yield dist_file

def median_distance_files_to_iter(aggregated_tweets_dir):
    for path, dirs, files in os.walk(aggregated_tweets_dir):
        for user_topics_dir in dirs:
            yield path + user_topics_dir, path + user_topics_dir + '/community_median_distances'
        break

def perform_clique_ops(lda, dictionary, output_dir, working_dir, lemmatize, cliq_top_file):
    # moves all the tweets of each clique into a single document
    # compares each of those documents to the LDA model to get the topic probability distribution
    # using each cliques aggregated tweets as one document.
    df = pd.read_csv(cliq_top_file, sep='\t', header=None)
    for idx, row in df.iterrows():
        aggregate_tweets(idx, row[0], 'dnld_tweets/', output_dir)

    # stores all of the aggregated tweet document vectors in a file, useful for plotting later
    try:
        with open(output_dir + working_dir + 'all_clique_doc_vecs.json', 'r') as infile:
            clique_vecs = json.load(infile)
    except:
        clique_vecs = {}

    # get the document vectors for the aggregated clique tweets by comparing those documents to an LDA model
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    func = partial(tlda.get_document_vectors, 
                   tweets_dir=output_dir, 
                   document_vectors=clique_vecs, 
                   dictionary=dictionary, 
                   lda_model=lda,
                   lemma=lemmatize) 
    doc_vecs = pool.map(func, cliques_to_iter(output_dir))
    doc_vecs = [item for item in doc_vecs if item is not None]
    clique_vecs.update(dict(doc_vecs))

    with open(output_dir + working_dir + 'all_clique_doc_vecs.json', 'w') as outfile:
        json.dump(clique_vecs, outfile, sort_keys=True, indent=4)
 
    # draws topic distribution graphs for the document vectors calculated from aggregated tweets of each clique
    func = partial(draw_dist_graph, 
                   output_dir=(output_dir + working_dir + 'distribution_graphs/'), 
                   doc_vecs=clique_vecs)
    pool.map(func, cliques_to_iter(output_dir))

    return clique_vecs
   
def perform_community_ops(lda, dictionary, distance_dir, lemmatize, clique_vecs, comm_top_file):
    # get the saved document vectors from the communities
    # and calculate the JSD between the each community user to the clique the community was derived from
    # 
    # this is a comparison of the topic distribution of a single users tweets in a community to the 
    # topic distribution of an entire cliques aggregated tweets
    try:
        with open(working_dir + 'document_vectors.json', 'r') as infile:
            all_community_doc_vecs = json.load(infile)
    except:
        all_community_doc_vecs = {}

    with open(comm_top_file, 'r') as topology:
        for i, community in enumerate(topology):
            func = partial(tlda.get_document_vectors, 
                           tweets_dir='dnld_tweets/', 
                           document_vectors=all_community_doc_vecs, 
                           dictionary=dictionary, 
                           lda_model=lda,
                           lemma=lemmatize)
            doc_vecs = pool.map(func, [str(user) for user in ast.literal_eval(community)])
            doc_vecs = [item for item in doc_vecs if item is not None]
            doc_vecs = dict(doc_vecs)

            print('Writing Jensen Shannon divergences for community ' + str(i))
            with open(distance_dir + 'community_' + str(i), 'w') as outfile:
                for user in doc_vecs:
                    jsd = pltd.jensen_shannon_divergence(clique_vecs['clique_' + str(i)], doc_vecs[user])
                    outfile.write('{}\t{}\t{}\n'.format(user, 'clique', jsd))
    
def get_clique_size(cliq_top_file):
    clique_size = []
    df = pd.read_csv(cliq_top_file, sep='\t', header=None)
    for idx, row in df.iterrows():
        clique_size.append(len(ast.literal_eval(row[0])))
    return clique_size

def get_community_size(comm_top_file):
    community_size = []
    with open(comm_top_file, 'r') as topology:
        for i, community in enumerate(topology):
            community_size.append(len(ast.literal_eval(community)))
    return community_size

def main():
    parser = argparse.ArgumentParser(description='Plot distances between community users and the cliques they spawned from')
    parser.add_argument('-c', '--clique_topology_file', required=True, action='store', dest='cliq_top_file', help='Location of clique topology file')
    parser.add_argument('-q', '--cliques', action='store_true', dest='cliques', help='Perform operations for cliques')
    parser.add_argument('-y', '--community_topology_file', required=True, action='store', dest='comm_top_file', help='Location of community topology file')
    parser.add_argument('-z', '--comms', action='store_true', dest='comms', help='Perform operations for communities')
    parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location of the saved LDA model')
    parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary for the model')
    parser.add_argument('-m', '--lemma', action='store_true', dest='lemma', help='Use this option to lemmatize words')

    parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help="""Name of the directory you want to direct output to.
                                                                                                          If a working directory was created by previously running
                                                                                                          tweets_on_LDA.py, you can use the name of that directory
                                                                                                          and it will use the document_vectors.json file to speed
                                                                                                          up the process. Otherwise, all the distance vectors
                                                                                                          for each user must be computed.""")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    dictionary = corpora.Dictionary.load(args.dict_loc)
    lda = models.LdaModel.load(args.lda_loc)

    # creates output directories for graphs and figures
    output_dir = 'aggregated_tweets/'
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), 0o755)
    if not os.path.exists(os.path.dirname(output_dir + args.working_dir + 'distribution_graphs/')):
        os.makedirs(os.path.dirname(output_dir + args.working_dir + 'distribution_graphs/'), 0o755)
    if not os.path.exists(os.path.dirname(output_dir + args.working_dir + 'community_user_distances/')):
        os.makedirs(os.path.dirname(output_dir + args.working_dir + 'community_user_distances/'), 0o755)

    distance_dir = output_dir + working_dir + 'community_user_distances/'

    # writes the top n words of the LDA model 
    tlda.write_topn_words(output_dir + args.working_dir, lda)

    clique_size = get_clique_size(args.cliq_top_file)
    community_size = get_community_size(args.comm_top_file)

    if args.cliques:
        clique_vecs = perform_clique_ops(lda, dictionary, output_dir, args.working_dir, args.lemma, args.cliq_top_file)

    if args.comms and clique_vecs:
        perform_community_ops(lda, dictionary, distance_dir, args.lemma, clique_vecs, args.comm_top_file)

    # draw graph comparing user JSD to their cliques or communities 
    func = partial(draw_user_to_clique_graphs, distance_dir)
    pool.map(func, distance_files_to_iter(distance_dir))

    # write the median JSD of each community compared to the clique it was made from to file
    # and plot the data
    if clique_size and community_size:
        print('Writing median distance of communities away from cliques')
        dist_files = [dist for dist in os.listdir(distance_dir) if not '.png' in dist]
        for dist in dist_files:
            df = pd.read_csv(distance_dir + dist, sep='\t', header=None, names=['user', 'clique', 'distance'])
            distances = [float(row['distance']) for idx, row in df.iterrows() if row['distance']]
            if distances:
                cid = [int(cid) for cid in dist.split('_') if cid.isdigit()]
                with open(output_dir + args.working_dir + 'community_median_distances', 'a') as outfile:
                    outfile.write('{}\t{}\t{}\t{}\n'.format(dist, np.median(distances), clique_size[cid[0]], community_size[cid[0]])) 

        for user_topics_dir, distance_file in median_distance_files_to_iter(output_dir):
            df = pd.read_csv(distance_file, sep='\t', header=None, names=['comm_id', 'avg_distance', 'cliq_size', 'comm_size'])
            draw_community_median_distances(user_topics_dir, distance_file, df)

    pool.close()

if __name__ == '__main__':
    sys.exit(main())
