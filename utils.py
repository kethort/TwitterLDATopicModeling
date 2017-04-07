import copy
import gensim
from gensim import models
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

# http://nbviewer.jupyter.org/github/dsquareindia/gensim/blob/a4b2629c0fdb0a7932db24dfcf06699c928d112f/docs/notebooks/topic_coherence_tutorial.ipynb
def get_topic_coherence():
    dictionary = Dictionary.load('./data/author_topic.dict')
    corpus = MmCorpus('./data/author_topic.mm')
    lda = LdaModel.load('./data/at_100_lem_5_pass_2.model')
    cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print(cm.get_coherence())


def write_overall_average_divergence_per_model(user_topics_dir, lda_loc):
    # write overall internal & external average community distance for each topic model
    # file format: num_topics, avg_int_dist, avg_ext_dist
    print('Writing overall internal & external average community distance for each topic model')
    lda = models.LdaModel.load(lda_loc)
	clique_int_dists = []
    clique_ext_dists = []
    comm_int_dists = []
    comm_ext_dists = []
    fieldnames = ['metric', 'distance']
    for path, dirs, files in os.walk(user_topics_dir):
        for community in dirs:
            with open(path + community + '/distance_info/community_average_distances', 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                if 'clique' in community:
                    clique_int_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
                else:
                    comm_int_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]

            with open(path + community + '/distance_info/external_average_distances', 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                if 'clique' in community:
                    clique_ext_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
                else:
                    comm_ext_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
        break

    if 'twitter' in user_topics_dir:
        with open('twitter_num_topics_divergence', 'a') as outfile:
            outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(lda.num_topics, np.average(clique_int_dists), np.average(clique_ext_dists), np.average(comm_int_dists), np.average(comm_ext_dists)))
    else:
        with open('wiki_num_topics_divergence', 'a') as outfile:
            outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(lda.num_topics, np.average(clique_int_dists), np.average(clique_ext_dists), np.average(comm_int_dists), np.average(comm_ext_dists)))

def draw_num_topics_average_divergence(cliq_dict, comm_dict, output_name):
	x_axis = np.arange(25, 101, 25)
    int_y_axis = [cliq_dict[str(i)][0] for i in x_axis]
    ext_y_axis = [cliq_dict[str(i)][1] for i in x_axis]
    plt.figure(1)
    plt.suptitle('Average Overall Divergence for Number of Topics in Twitter Model', fontsize=14, fontweight='bold')
    plt.subplot(211)
    plt.plot(x_axis, int_y_axis, '-g^', x_axis, ext_y_axis, '-bo')
    plt.xlabel('Number of Topics')
    plt.ylabel('Jensen Shannon Divergence')
    plt.title('Clique')
    plt.xticks(x_axis, x_axis, fontsize=8)
    plt.yticks(fontsize=10)
    plt.ylim([0, np.log(2) + 0.01])
    plt.xlim([min(x_axis), max(x_axis)])

    int_y_axis = [comm_dict[str(i)][0] for i in x_axis]
    ext_y_axis = [comm_dict[str(i)][1] for i in x_axis]
    plt.subplot(212)
    ax = plt.plot(x_axis, int_y_axis, '-g^', x_axis, ext_y_axis, '-bo')
    plt.xlabel('Number of Topics')
    plt.ylabel('Jensen Shannon Divergence')
    plt.title('Community')
    plt.xticks(x_axis, x_axis, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim([0, np.log(2) + 0.01])
    plt.xlim([min(x_axis), max(x_axis)])

    plt.figlegend((ax[0], ax[1]), ('Internal', 'External'), loc='lower center', ncol=5, labelspacing=0. ) 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, top=0.9)
    plt.savefig(output_name)
    plt.close()

def draw_num_topics_average_divergence_helper():
    fieldnames = ['num_topics', 'cliq_int_dist', 'cliq_ext_dist', 'comm_int_dist', 'comm_ext_dist']
    with open('twitter_num_topics_divergence', 'r') as infile:
        csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
        twitter_clique = dict((row['num_topics'], [row['cliq_int_dist'], row['cliq_ext_dist']]) for row in csv_reader)
        infile.seek(0)
        twitter_comm = dict((row['num_topics'], [row['comm_int_dist'], row['comm_ext_dist']]) for row in csv_reader)

    with open('wiki_num_topics_divergence', 'r') as infile:
        csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
        wiki_clique = dict((row['num_topics'], [row['cliq_int_dist'], row['cliq_ext_dist']]) for row in csv_reader)
        infile.seek(0)
        wiki_comm = dict((row['num_topics'], [row['comm_int_dist'], row['comm_ext_dist']]) for row in csv_reader)

	draw_num_topics_average_divergence(twitter_clique, twitter_comm, 'twitter_num_topics_divergence')
	draw_num_topics_average_divergence(wiki_clique, wiki_comm, 'wiki_num_topics_divergence')

def user_tweet_distribution():
    """

    plots the distribution of the amount of tweets on every users Twitter timeline

    The amount of tweets is the number of lines in each users' tweet document. Each line in the users tweet
    document isn't always necessarily only one tweet. The maximum amount of tweets that can be downloaded through
    the API for each user is 3200. This graph is only an approximation of the distribution of the users' tweets.

    **this only needs to be run once for all tweets downloaded

    """
    with open('user_tweet_count.json', 'r') as infile:
        d = json.load(infile)

    x_axis = np.arange(0, 3300, 100)
    y_vals = get_y_vals(d, x_axis)

    plt.bar(x_axis, y_vals, align='center', width=80, color='y')
    plt.xlabel('Number of Tweets')
    plt.ylabel('Number of Users')
    plt.title('Tweets per User ')
    plt.xticks(x_axis, x_axis, rotation='vertical', fontsize='small')
    plt.ylim([0, np.max(y_vals)])
    plt.xlim([0, np.max(x_axis)])
    plt.savefig('tweet_distribution')
    plt.close()

def get_y_vals(d, x_axis):
    y_vals = [0] * len(x_axis)

    for i in range(0, len(x_axis)):
        for key in d:
            if(i == len(x_axis) - 1):
                if(d[key] >= x_axis[i]):
                    y_vals[i] += 1
            else:
                if(d[key] >= x_axis[i] and d[key] < x_axis[i + 1]):
                    y_vals[i] += 1
    return y_vals

def community_size_distribution():
    """

    plots the distribution of the sizes of all communities and cliques in the dataset

    **this only needs to be run once and will be the same no matter which user topics
      directory is used

    """
    sizes = {}
    with open('cliques', 'r') as infile:
        for i, line in enumerate(infile):
            sizes['clique_' + str(i)] = len(ast.literal_eval(line))
    
    with open('communities', 'r') as infile:
        for i, line in enumerate(infile):
            sizes['community_' + str(i)] = len(ast.literal_eval(line))

    with open('cliq_comm_sizes.json', 'w') as outfile:
        json.dump(sizes, outfile, sort_keys=True, indent=4)

    x_axis = np.arange(0, sizes[max(sizes, key=lambda i: sizes[i])], 10)
    y_vals = get_y_vals(sizes, x_axis)

    plt.bar(x_axis, y_vals, align='center', width=10, color='y')
    plt.xlabel('Community Size Distribution')
    plt.ylabel('Number of Communities')
    plt.title('Community Size Distribution')
    plt.xticks(x_axis, x_axis, rotation='vertical', fontsize='small')
    plt.ylim([0, np.max(y_vals)])
    plt.xlim([0, np.max(x_axis)])
    plt.savefig('community_size_distribution')
    plt.close()

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

def write_tweet_meta(tweets, meta_filename, followers_filename):
    ''' 
        writes the Tweet metadata being scraped to a file as:
        tweet_type, user_id, RT_user_id, RT_count, tweet_id, hashtags, screen_name
    '''
    with open(meta_filename, 'a') as clique_tweet_metadata:
        for tweet in tweets:
            user_followers = {}
            favorite_count = tweet.favorite_count
            tweet_id = tweet.id_str
            screen_name = tweet.user.screen_name
            retweet_count = tweet.retweet_count
            user_id = tweet.user.id
            follower_count = tweet.user.followers_count
        
            if os.path.exists(followers_filename):
                with open(followers_filename, 'r') as follower_dump:
                    user_followers = json.load(follower_dump)

            # get the follower count of each user
            if not any(str(user_id) in key for key in user_followers):
                user_followers[str(user_id)] = str(follower_count)
            
            # serialize dictionary to save memory
            with open(followers_filename, 'w') as follower_dump:
                json.dump(user_followers, follower_dump)
                
            user_followers = {}

            # extract hashtags
            tagList = tweet.entities.get('hashtags')
            # check if there are hashtags
            if(len(tagList) > 0):
                hashtags = [tag['text'] for tag in tagList]
        
            # if the tweet is not a retweet
            if not hasattr(tweet, 'retweeted_status'):
                out = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('T', user_id, user_id, retweet_count, tweet_id, hashtags, screen_name) 
            # if it is retweet, get user id of original tweet 
            else:
                rt_user_id = tweet.retweeted_status.user.id
                rt_screen_name = tweet.retweeted_status.user.screen_name
                orig_tweet_id = tweet.retweeted_status.id_str
        
                out = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('RT', user_id, rt_user_id, retweet_count, orig_tweet_id, hashtags, rt_screen_name) 
            clique_tweet_metadata.write(out)

