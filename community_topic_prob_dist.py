# -*- coding: utf-8 -*-
import gensim
from gensim import utils, corpora, models
import csv
import ast
import re
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import tweets_on_LDA as tlda
import plot_distances as pltd
import multiprocessing
from functools import partial

def aggregate_tweets(i, clique, tweets_dir, output_dir):
    if not os.path.exists(output_dir + 'clique_' + str(i)):
        print('Aggregating tweets for clique_' + str(i))
        with open(output_dir + 'clique_' + str(i), 'w') as outfile:
            for user in ast.literal_eval(clique):
                if os.path.exists(tweets_dir + str(user)):
                    with open(tweets_dir + str(user)) as tweet:
                        for line in tweet:
                            outfile.write(str(line))

def draw_dist_graph(inputs, clique_name):
    output_dir = inputs[0]
    doc_vecs = inputs[1]

    if not os.path.exists(output_dir + clique_name + '.png'):
        print('Drawing probability distribution graph for ' + clique_name)
        y_axis = []
        x_axis = []
                        
        for topic_id, dist in enumerate(doc_vecs[clique_name]):
            x_axis.append(topic_id + 1)
            y_axis.append(dist)

        plt.bar(x_axis, y_axis, width=1, align='center', color='r')
        plt.xlabel('Topics')
        plt.ylabel('Probability')
        plt.title('Topic Distribution for clique')
        plt.xticks(np.arange(2, len(x_axis), 2), rotation='vertical', fontsize=7)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.max(y_axis) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(output_dir + clique_name)
        plt.close()

def draw_user_to_clique_graphs(distance_dir, dist_file):
    x_axis = []
    y_axis = []
    fieldnames = ['user', 'clique', 'distance']

    if not os.path.exists(distance_dir + dist_file + '.png'):
        print('Drawing users to clique for: ' + dist_file)
        with open(distance_dir + dist_file, 'r') as infile:
            csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
            for row in csv_reader:
                x_axis.append(str(row['user']))
                y_axis.append(float(row['distance']))

        plt.bar(np.arange(1, len(x_axis) + 1, 1), y_axis, width=1, align='center', color='r')
        #plt.plot(np.arange(1, len(x_axis) + 1, 1), y_axis, 'o', color='r')
        plt.xlabel('Community Users')
        plt.ylabel('Divergence from Clique')
        plt.title('Users to Clique ' + dist_file, fontsize=14, fontweight='bold')
        plt.xticks(np.arange(1, len(x_axis) + 1, 1), x_axis, rotation='vertical', fontsize=7)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.log(2) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(distance_dir + dist_file)
        plt.close()

def cliques_to_iter(tweet_folder):
    for path, dirs, files in os.walk(tweet_folder):
        for filename in files:   
            yield filename
        break # stop before traversing into newly created dirs

def distance_files_to_iter(distance_dir):
    for dist_file in os.listdir(distance_dir):
        if not dist_file.endswith('.png'):
            yield dist_file

# clique_top: clique topology, comm_top: community topology, tweets_dir: path of downloaded tweets dir
# dict_loc: dictionary, lda_loc: lda model,
# user_topics_dir: directory where lda model was used in plot_distances to create graphs

# python2.7 community_topic_prob_dist.py cliques communities dnld_tweets/ data/twitter/tweets.dict data/twitter/tweets_100_lda_lem_5_pass.model user_topics_100
def main(clique_top, comm_top, tweets_dir, dict_loc, lda_loc, user_topics_dir):
    # load wiki dictionary
    dictionary = corpora.Dictionary.load(dict_loc)

    # load trained wiki model from file
    lda = models.LdaModel.load(lda_loc)

    output_dir = 'aggregated_tweets_2/'

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), 0o755)

    if not os.path.exists(os.path.dirname(output_dir + user_topics_dir + 'distribution_graphs/')):
        os.makedirs(os.path.dirname(output_dir + user_topics_dir + 'distribution_graphs/'), 0o755)

    if not os.path.exists(os.path.dirname(output_dir + user_topics_dir + 'community_user_distances/')):
        os.makedirs(os.path.dirname(output_dir + user_topics_dir + 'community_user_distances/'), 0o755)

    tlda.write_topn_words(output_dir + user_topics_dir, lda)

    with open(clique_top, 'r') as infile:
        for i, clique in enumerate(infile):
            aggregate_tweets(i, clique, tweets_dir, output_dir)

    try:
        with open(output_dir + user_topics_dir + 'all_clique_doc_vecs.json', 'r') as infile:
            clique_vecs = json.load(infile)
    except:
        clique_vecs = {}

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    func = partial(tlda.get_document_vectors, (output_dir, clique_vecs, dictionary, lda)) 
    doc_vecs = pool.map(func, cliques_to_iter(output_dir))
    doc_vecs = [item for item in doc_vecs if item is not None]
    clique_vecs.update(dict(doc_vecs))

    with open(output_dir + user_topics_dir + 'document_vectors.json', 'w') as outfile:
        json.dump(clique_vecs, outfile, sort_keys=True, indent=4)
 
    func = partial(draw_dist_graph, (output_dir + user_topics_dir + 'distribution_graphs/', clique_vecs))
    pool.map(func, cliques_to_iter(output_dir))

    try:
        with open(user_topics_dir + 'all_community_doc_vecs.json', 'r') as infile:
            all_community_doc_vecs = json.load(infile)
    except:
        all_community_doc_vecs = {}

    with open(comm_top, 'r') as topology:
        for i, community in enumerate(topology):
            func = partial(tlda.get_document_vectors, (tweets_dir, all_community_doc_vecs, dictionary, lda))
            doc_vecs = pool.map(func, tlda.users_to_iter(community))
            doc_vecs = [item for item in doc_vecs if item is not None]
            doc_vecs = dict(doc_vecs)

            print('Writing Jensen Shannon divergences for community ' + str(i))
            with open(output_dir + user_topics_dir + 'community_user_distances/community_' + str(i), 'w') as outfile:
                for user in doc_vecs:
                    jsd = pltd.jensen_shannon_divergence(clique_vecs['clique_' + str(i)], doc_vecs[user])
                    outfile.write('{}\t{}\t{}\n'.format(user, 'clique', jsd))

    distance_dir = output_dir + user_topics_dir + 'community_user_distances/'
    func = partial(draw_user_to_clique_graphs, distance_dir)
    pool.map(func, distance_files_to_iter(distance_dir))

    pool.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))


