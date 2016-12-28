# -*- coding: utf-8 -*-
import time
import csv
import ast
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tweets_on_LDA as tlda
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
                        
        try:
            doc_vector = doc_vecs[clique_name]
        except KeyError:
            return

        for topic_id, dist in enumerate(doc_vector):
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

        plt.figure(figsize=(20, 10))
        plt.bar(np.arange(1, len(x_axis) + 1, 1), y_axis, width=1, align='center', color='r')
        plt.xlabel('Community Users')
        plt.ylabel('Divergence from Clique')
        plt.title('Users to Clique ' + dist_file, fontsize=14, fontweight='bold')
        plt.xticks(np.arange(0, len(x_axis) + 1, 2), np.arange(0, len(x_axis) + 1, 2), rotation='vertical', fontsize=8)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.log(2) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(distance_dir + dist_file)
        plt.close()

def draw_community_average_distances(user_topics_dir, distance_file, dist_dict):
    x_axis = np.arange(1, len(dist_dict) - 1)
    y_axis = [dist_dict['community_' + str(i)] for i in range(len(dist_dict)) if 'community_' + str(i) in dist_dict]

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, len(x_axis) + 1), y_axis, 'r')
    plt.xlabel('Community ID')
    plt.ylabel('Divergence from Clique')
    plt.title('Community Users Divergence from Clique', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(0, len(x_axis) + 1, 20), np.arange(0, len(x_axis) + 1, 20), rotation='vertical', fontsize=8)
    plt.subplots_adjust(bottom=0.2)
    plt.ylim([0, np.log(2) + 0.01])
    plt.xlim([0, len(x_axis) + 1])
    plt.savefig(distance_file)
    plt.close()

def cliques_to_iter(tweet_folder):
    for path, dirs, files in os.walk(tweet_folder):
        files.sort()
        for filename in files:   
            yield filename
        break # stop before traversing into newly created dirs

def distance_files_to_iter(distance_dir):
    for dist_file in os.listdir(distance_dir):
        if not dist_file.endswith('.png'):
            yield dist_file

def average_distance_files_to_iter(aggregated_tweets_dir):
    for path, dirs, files in os.walk(aggregated_tweets_dir):
        for user_topics_dir in dirs:
            yield path + user_topics_dir, path + user_topics_dir + '/community_average_distances'
        break

# clique_top: clique topology, comm_top: community topology, tweets_dir: path of downloaded tweets dir
# dict_loc: dictionary, lda_loc: lda model,
# user_topics_dir: directory where lda model was used in plot_distances to create graphs

# python2.7 community_topic_prob_dist.py cliques communities dnld_tweets/ data/twitter/tweets.dict data/twitter/tweets_100_lda_lem_5_pass.model user_topics_100
def main(clique_top, comm_top, tweets_dir, dict_loc, lda_loc, user_topics_dir):
    start_time = time.time()
    # load wiki dictionary
    dictionary = corpora.Dictionary.load(dict_loc)

    # load trained wiki model from file
    lda = models.LdaModel.load(lda_loc)

    output_dir = 'aggregated_tweets/'

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), 0o755)

    if not os.path.exists(os.path.dirname(output_dir + user_topics_dir + 'distribution_graphs/')):
        os.makedirs(os.path.dirname(output_dir + user_topics_dir + 'distribution_graphs/'), 0o755)

    if not os.path.exists(os.path.dirname(output_dir + user_topics_dir + 'community_user_distances/')):
        os.makedirs(os.path.dirname(output_dir + user_topics_dir + 'community_user_distances/'), 0o755)

    tlda.write_topn_words(output_dir + user_topics_dir, lda)

    clique_size = []

    with open(clique_top, 'r') as infile:
        for i, clique in enumerate(infile):
            aggregate_tweets(i, clique, tweets_dir, output_dir
            clique_size.append(len(ast.literal_eval(clique)))

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

    with open(output_dir + user_topics_dir + 'all_clique_doc_vecs.json', 'w') as outfile:
        json.dump(clique_vecs, outfile, sort_keys=True, indent=4)
 
    func = partial(draw_dist_graph, (output_dir + user_topics_dir + 'distribution_graphs/', clique_vecs))
    pool.map(func, cliques_to_iter(output_dir))

    try:
        with open(user_topics_dir + 'all_community_doc_vecs.json', 'r') as infile:
            all_community_doc_vecs = json.load(infile)
    except:
        all_community_doc_vecs = {}

    community_size = []

    with open(comm_top, 'r') as topology:
        for i, community in enumerate(topology):
            community_size.append(len(ast.literal_eval(community)))

            func = partial(tlda.get_document_vectors, (tweets_dir, all_community_doc_vecs, dictionary, lda))
            doc_vecs = pool.map(func, tlda.users_to_iter(community))
            doc_vecs = [item for item in doc_vecs if item is not None]
            doc_vecs = dict(doc_vecs)

            print('Writing Jensen Shannon divergences for community ' + str(i))
            with open(output_dir + user_topics_dir + 'community_user_distances/community_' + str(i), 'w') as outfile:
                for user in doc_vecs:
                    try:
                        jsd = pltd.jensen_shannon_divergence(clique_vecs['clique_' + str(i)], doc_vecs[user])
                        outfile.write('{}\t{}\t{}\n'.format(user, 'clique', jsd))
                    except:
                        continue

    distance_dir = output_dir + user_topics_dir + 'community_user_distances/'
    func = partial(draw_user_to_clique_graphs, distance_dir)
    pool.map(func, distance_files_to_iter(distance_dir))

   # write average distance of community away from clique 
   # file format: community_id, average_distance, community_size, clique_size
    print('Writing average distance of communities away from cliques')
    fieldnames = ['user', 'clique', 'distance']
    dist_files = [dist for dist in os.listdir(output_dir + user_topics_dir + 'community_user_distances/') if not '.png' in dist]
    for dist in dist_files:
        with open(output_dir + user_topics_dir + 'community_user_distances/' + dist, 'r') as infile:
            csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
            distances = [float(row['distance']) for row in csv_reader if row['distance']]
        if distances:
            cid = [int(cid) for cid in dist.split('_') if cid.isdigit()]
            with open(output_dir + user_topics_dir + 'community_average_distances', 'a') as outfile:
                outfile.write('{}\t{}\t{}\t{}\n'.format(dist, np.average(distances), clique_size[cid[0]], community_size[cid[0]])) 


    fieldnames = ['comm_id', 'avg_distance', 'comm_size', 'cliq_size']
    for user_topics_dir, distance_file in average_distance_files_to_iter(output_dir):
        with open(distance_file, 'r') as infile:
            csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
            draw_community_average_distances(user_topics_dir, distance_file, dict((row['comm_id'], row['avg_distance']) for row in csv_reader))

    pool.close()
    print('Runtime: ' + str((((time.time() - start_time) / 60) / 60)) + ' hours')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))


