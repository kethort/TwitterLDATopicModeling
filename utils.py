import copy
import gensim
import shelve
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import ast
import sys
import random
import click
import csv
import pickle
import shutil
from shutil import copyfile

import warnings
warnings.filterwarnings('ignore')

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import MmCorpus, Dictionary
from gensim.models import VocabTransform

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def filter_saved_corpus():
    # Q8 - https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ
    DEFAULT_DICT_SIZE = 100000

    # filter the dictionary
    old_dict = gensim.corpora.Dictionary.load('data.new_old/wiki_dict.dict')
    new_dict = copy.deepcopy(old_dict)
    new_dict.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)
    new_dict.save('data.new_old/filtered.dict')

    # transform the corpus
    corpus = gensim.corpora.MmCorpus('data.new_old/wiki_corpus.mm')
    old2new = {old_dict.token2id[token]:new_id for new_id, token in new_dict.iteritems()}
    vt = VocabTransform(old2new)
    gensim.corpora.MmCorpus.serialize('data.new_old/filtered_corpus.mm', vt[corpus], id2word=new_dict, progress_cnt=10000)

    # create lda model from filtered data
    bow_corpus = gensim.corpora.MmCorpus('data.new_old/filtered_corpus.mm')
    dictionary = gensim.corpora.Dictionary.load('data.new_old/filtered.dict')
    lda = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)
    lda.save('data.new_old/lda_filtered.model')

def get_topic_coherence():
	dictionary = Dictionary.load('./data/author_topic.dict')
        corpus = MmCorpus('./data/author_topic.mm')
	lda = LdaModel.load('./data/at_100_lem_5_pass_2.model')
	cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
	print(cm.get_coherence())

def user_tweet_distribution():
    """

    plots the distribution of the amount of tweets on every users Twitter timeline

    The amount of tweets is the number of lines in each users' tweet document. Each line in the users tweet
    document isn't always necessarily only one tweet. The maximum amount of tweets that can be downloaded through
    the API for each user is 3200. This graph is only an approximation of the distribution of the users' tweets.

    **this only needs to be run once for all tweets downloaded

    """
    objects = np.arange(0, 5100, 100)
    values = [0] * len(objects)
    max_count = 0
    progress_label = 'Getting user tweet distribution: '
    with click.progressbar(os.listdir('tweets_dir/'), label=progress_label) as bar:
        for tweet in bar:
            count = 0
            with open('tweets_dir/' + tweet, 'r') as infile:
                for line in infile:
                    if line.strip():
                        count += 1

            if count >= 0 and count <= 100:
                values[0] += 1
            elif count > 100 and count <= 200:
                values[1] += 1
            elif count > 200 and count <= 300:
                values[2] += 1
            elif count > 300 and count <= 400:
                values[3] += 1
            elif count > 400 and count <= 500:
                values[4] += 1
            elif count > 500 and count <= 600:
                values[5] += 1
            elif count > 600 and count <= 700:
                values[6] += 1
            elif count > 700 and count <= 800:
                values[7] += 1
            elif count > 800 and count <= 900:
                values[8] += 1
            elif count > 900 and count <= 1000:
                values[9] += 1
            elif count > 1000 and count <= 1100:
                values[10] += 1
            elif count > 1100 and count <= 1200:
                values[11] += 1
            elif count > 1200 and count <= 1300:
                values[12] += 1
            elif count > 1300 and count <= 1400:
                values[13] += 1
            elif count > 1400 and count <= 1500:
                values[14] += 1
            elif count > 1500 and count <= 1600:
                values[15] += 1
            elif count > 1600 and count <= 1700:
                values[16] += 1
            elif count > 1700 and count <= 1800:
                values[17] += 1
            elif count > 1800 and count <= 1900:
                values[18] += 1
            elif count > 1900 and count <= 2000:
                values[19] += 1
            elif count > 2000 and count <= 2100:
                values[20] += 1
            elif count > 2100 and count <= 2200:
                values[21] += 1
            elif count > 2200 and count <= 2300:
                values[22] += 1
            elif count > 2300 and count <= 2400:
                values[23] += 1
            elif count > 2400 and count <= 2500:
                values[24] += 1
            elif count > 2500 and count <= 2600:
                values[25] += 1
            elif count > 2600 and count <= 2700:
                values[26] += 1
            elif count > 2700 and count <= 2800:
                values[27] += 1
            elif count > 2800 and count <= 2900:
                values[28] += 1
            elif count > 2900 and count <= 3000:
                values[29] += 1
            elif count > 3000 and count <= 3100:
                values[30] += 1
            elif count > 3100 and count <= 3200:
                values[31] += 1
            elif count > 3200 and count <= 3300:
                values[32] += 1
            elif count > 3300 and count <= 3400:
                values[33] += 1
            elif count > 3400 and count <= 3500:
                values[34] += 1
            elif count > 3500 and count <= 3600:
                values[35] += 1
            elif count > 3600 and count <= 3700:
                values[36] += 1
            elif count > 3700 and count <= 3800:
                values[37] += 1
            elif count > 3800 and count <= 3900:
                values[38] += 1
            elif count > 3900 and count <= 4000:
                values[39] += 1
            elif count > 4000 and count <= 4100:
                values[40] += 1
            elif count > 4100 and count <= 4200:
                values[41] += 1
            elif count > 4200 and count <= 4300:
                values[42] += 1
            elif count > 4300 and count <= 4400:
                values[43] += 1
            elif count > 4400 and count <= 4500:
                values[44] += 1
            elif count > 4500 and count <= 4600:
                values[45] += 1
            elif count > 4600 and count <= 4700:
                values[46] += 1
            elif count > 4700 and count <= 4800:
                values[47] += 1
            elif count > 4800 and count <= 4900:
                values[48] += 1
            elif count > 4900 and count <= 5000:
                values[49] += 1
            elif count > 5000:
                values[50] += 1

    x_axis = np.arange(0, len(objects) * 2, 2)
    plt.bar(x_axis + 0.2, values, align='center', width=0.8, color='b')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Number of Users')
    plt.title('Tweets per User ')
    plt.xticks(x_axis, objects, rotation='vertical', fontsize='small')
    plt.ylim([0, np.max(values) + 1])
    plt.xlim([0, (len(objects) * 2) + 1])
    plt.savefig('tweet_distribution')
    plt.close()

def community_size_distribution():
    """

    plots the distribution of the sizes of all communities and cliques in the dataset

    **this only needs to be run once and will be the same no matter which user topics
      directory is used

    """

    objects = np.arange(0, 410, 10)
    comm_dirs = []
    sizes = []
    values = [0] * len(objects)
    for path, dirs, files in os.walk('user_topics_100/'):
        for community in dirs:
            comm_dirs.append(path + community)
        break

    for community in comm_dirs:
        try:
            with open(community + '/community_doc_vecs.pickle', 'rb') as comm_doc_vecs_file:
                comm_doc_vecs = pickle.load(comm_doc_vecs_file)
        except:
            comm_doc_vecs = {}

        size = len(comm_doc_vecs)
        if(size == 0):
            continue
        sizes.append(size)

        if(size > 0 and size <= 10):
            values[0] += 1
        elif(size > 10 and size <= 20):
            values[1] += 1
        elif(size > 20 and size <= 30):
            values[2] += 1
        elif(size > 30 and size <= 40):
            values[3] += 1
        elif(size > 40 and size <= 50):
            values[4] += 1
        elif(size > 50 and size <= 60):
            values[5] += 1
        elif(size > 60 and size <= 70):
            values[6] += 1
        elif(size > 70 and size <= 80):
            values[7] += 1
        elif(size > 80 and size <= 90):
            values[8] += 1
        elif(size > 90 and size <= 100):
            values[9] += 1
        elif(size > 100 and size <= 110):
            values[10] += 1
        elif(size > 110 and size <= 120):
            values[11] += 1
        elif(size > 120 and size <= 130):
            values[12] += 1
        elif(size > 130 and size <= 140):
            values[13] += 1
        elif(size > 140 and size <= 150):
            values[14] += 1
        elif(size > 150 and size <= 160):
            values[15] += 1
        elif(size > 160 and size <= 170):
            values[16] += 1
        elif(size > 170 and size <= 180):
            values[17] += 1
        elif(size > 180 and size <= 190):
            values[18] += 1
        elif(size > 190 and size <= 200):
            values[19] += 1
        elif(size > 200 and size <= 210):
            values[20] += 1
        elif(size > 210 and size <= 220):
            values[21] += 1
        elif(size > 220 and size <= 230):
            values[22] += 1
        elif(size > 230 and size <= 240):
            values[23] += 1
        elif(size > 240 and size <= 250):
            values[24] += 1
        elif(size > 250 and size <= 260):
            values[25] += 1
        elif(size > 260 and size <= 270):
            values[26] += 1
        elif(size > 270 and size <= 280):
            values[27] += 1
        elif(size > 280 and size <= 290):
            values[28] += 1
        elif(size > 290 and size <= 300):
            values[29] += 1
        elif(size > 300 and size <= 310):
            values[30] += 1
        elif(size > 310 and size <= 320):
            values[31] += 1
        elif(size > 320 and size <= 330):
            values[32] += 1
        elif(size > 330 and size <= 340):
            values[33] += 1
        elif(size > 340 and size <= 350):
            values[34] += 1
        elif(size > 350 and size <= 360):
            values[35] += 1
        elif(size > 360 and size <= 370):
            values[36] += 1
        elif(size > 370 and size <= 380):
            values[37] += 1
        elif(size > 380 and size <= 390):
            values[38] += 1
        elif(size > 390 and size <= 400):
            values[39] += 1
        elif(size > 400):
            values[40] += 1

    x_axis = np.arange(0, len(objects) * 2, 2)
    plt.bar(x_axis, values, align='center', width=0.8, color='b')
    plt.xlabel('Community Size Distribution')
    plt.ylabel('Number of Cliques/Communities')
    plt.title('Community Size Distribution')
    plt.xticks(x_axis, objects, rotation='vertical', fontsize='small')
    plt.ylim([0, np.max(values) + 1])
    plt.xlim([0, (len(objects) * 2) + 1])
    plt.savefig('community_size_distribution')
    plt.close()

def aggregate_clq_comm_tweets():
    """

    aggregates each of the users tweets in a community into one
    community tweet document

    """
    agg_tweet_dir = 'aggregated_tweets/'
    if not os.path.exists(os.path.dirname(agg_tweet_dir)):
        os.makedirs(os.path.dirname(agg_tweet_dir), 0o755)

    for path, dirs, files in os.walk('tweets/'):
        for community in sorted(dirs):
            print('Aggregating tweets for community: ' + str(community))
            with open(agg_tweet_dir + community, 'w') as outfile:
                for tweet in os.listdir(path + community):
                    with open(path + community + '/' + tweet, 'r') as infile:
                        for line in infile:
                            if line.strip():
                                outfile.write(line)
        break

def get_aggregated_doc_vectors():
    """

    obtains topic distribution document vectors for aggregated community tweets

    """
    all_comm_vecs = {}
    dictionary = corpora.Dictionary.load('data/wiki_dict.dict')
    lda = models.LdaModel.load('data/lda_100_lem_5_pass.model')

    for tweets in os.listdir('aggregated_tweets/'):
        print('Getting topic distribution vector for: ' + str(tweets))
        doc = tlda.convert_to_doc('aggregated_tweets/' + tweets)
        doc_bow = dictionary.doc2bow(doc)
        doc_lda = tlda.get_doc_topics(lda, doc_bow)
        dense_vec = matutils.sparse2full(doc_lda, lda.num_topics)
        all_comm_vecs[tweets] = dense_vec

    with open('aggregated_tweets/lda_100_doc_vecs.pickle', 'wb') as outfile:
        pickle.dump(all_comm_vecs, outfile)

def revert_to_downloaded_state():
    """

    places all the tweets that were previously ommitted or preprocessed from the data
    back into the folders that they were in when they were downloaded using the 
    get_community_tweets.py script
    
    this method expects that both the cliques topology file and the communities topology
    file is in the same folder as the plot_distances.py script and that the 
    tweets_dir/ folder created from the remove_users_less_than_n_tweets() method exists
    
    """

    with open('cliques') as infile:
        for i, clique in enumerate(infile):
            if not os.path.exists(os.path.dirname('tweets/clique_' + str(i) + '/')):
                os.makedirs(os.path.dirname('tweets/clique_' + str(i) + '/'), 0o755)
            for user in ast.literal_eval(clique):
                if not os.path.exists('tweets/clique_' + str(i) + '/' + str(user)) and os.path.exists('tweets_dir/' + str(user)):
                    copyfile('tweets_dir/' + str(user), 'tweets/clique_' + str(i) + '/' + str(user))
                else:
                    continue

    with open('communities') as infile:
        for i, community in enumerate(infile):
            if not os.path.exists(os.path.dirname('./tweets/community_' + str(i) + '/')):
                os.makedirs(os.path.dirname('./tweets/community_' + str(i) + '/'), 0o755)
            for user in ast.literal_eval(community):
                if not os.path.exists('tweets/community_' + str(i) + '/' + str(user)) and os.path.exists('tweets_dir/' + str(user)):
                    copyfile('./tweets_dir/' + str(user), './tweets/community_' + str(i) + '/' + str(user))
                else:
                    continue
def delete_old_figures(user_topics_dir):
    """

    deletes all the graph data in each community directory
    keeping only the serialized document vectors

    don't run this unless you have a backup, lots of data will be lost
    """
    for path, dirs, files in os.walk(user_topics_dir):
        for community in sorted(dirs):
            if os.path.exists(path + community + '/distance_difference_graphs/'):
                shutil.rmtree(path + community + '/distance_difference_graphs/')
            if os.path.exists(path + community + '/topic_distribution_graphs/'):
                shutil.rmtree(path + community + '/topic_distribution_graphs/')
            if os.path.exists(path + community + '/distance_info/'):
                shutil.rmtree(path + community + '/distance_info/')
            if os.path.exists(path + community + '/user_to_internal_users_graphs/'):
                shutil.rmtree(path + community + '/user_to_internal_users_graphs/')
            if os.path.exists(path + community + '/user_to_external_users_graphs/'):
                shutil.rmtree(path + community + '/user_to_external_users_graphs/')
            if os.path.exists(path + community + '/num_users_distance_range_graphs/'):
                shutil.rmtree(path + community + '/num_users_distance_range_graphs/')
        break

def convert_pickle_to_json(user_topics_dir, community):
    """

    converts the serialized community document vectors into
    a human-readable format

    """

    if not os.path.exists(user_topics_dir + '/all_community_doc_vecs.json'):
        with open(user_topics_dir + '/all_community_doc_vecs.pickle', 'rb') as pickle_dump:
            all_comm_doc_vecs = pickle.load(pickle_dump)
        for user in all_comm_doc_vecs:
            all_comm_doc_vecs[user] = all_comm_doc_vecs[user].tolist()
        with open(user_topics_dir + '/all_community_doc_vecs.json', 'w') as json_dump:
            json.dump(all_comm_doc_vecs, json_dump, sort_keys=True, indent=4)


    with open(community + '/community_doc_vecs.pickle', 'rb') as pickle_dump:
        comm_doc_vecs = pickle.load(pickle_dump)

    progress_label = 'Converting pickle to json for: ' + str(community)
    with click.progressbar(comm_doc_vecs, label=progress_label) as doc_vectors:
        for user in doc_vectors:
            comm_doc_vecs[user] = comm_doc_vecs[user].tolist()
        with open(community + '/community_doc_vecs.json', 'w') as json_dump:
            json.dump(comm_doc_vecs, json_dump, sort_keys=True, indent=4)

# converts the entire topology file into an on-disk queryable dictionary
# database file in the format: (user_id, list_of_followers[])
def convert_topology_to_shelf():
    # stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    with open('topology.txt', 'r') as topfile:
        buf = mmap.mmap(topfile.fileno(), 0, prot=mmap.PROT_READ)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
    tmp = '' 
    fieldnames = ['user', 'follower']
    progress_label = 'Converting topology.txt to shelf database file'
    with open('topology.txt', 'r') as topfile:
        csv_reader = csv.DictReader(topfile, delimiter=',', fieldnames=fieldnames)
        with click.progressbar(csv_reader, label=progress_label, length=lines) as bar:
            for row in bar:
            # create a dictionary from topology file as (user, followers[])<k,v>
                # if a new user is found in the topology, empty the list
                if tmp != row['user']:
                    if tmp:
                        shelf = shelve.open('topology.shelf', 'c')
                        # save the (user, follower[]) in shelf
                        shelf[tmp] = follower_list
                        shelf.close()
                    follower_list = []
                follower_list.append(row['follower'])
                # keep track of current user
                tmp = row['user']


