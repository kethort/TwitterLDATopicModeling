# -*- coding: utf-8 -*-
import get_community_tweets
import subprocess
import click
import shutil
from shutil import copyfile
import bz2
import csv
import random
import numpy as np
import scipy
from scipy.spatial import distance
from scipy.linalg import norm
from scipy.stats import entropy
import textblob
import gensim
import logging
import itertools
from collections import defaultdict
from pprint import pprint
import pickle
import sys
import mmap
from contextlib import closing
import shelve
import re
import os
from os import path
from gensim import utils, corpora, models, similarities

# for each user document vector find the distance from every other user document vector
# in the community. Dictionary <k, v>(user_id, doc_vec)
def community_user_distances(community_dir):
    cos_file = community_dir + '/distance_info/cosine'  
    hell_file = community_dir + '/distance_info/hellinger'
    euc_file = community_dir + '/distance_info/euclidean'
    jen_shan_file = community_dir + '/distance_info/jensen_shannon'

    outfiles = [cos_file, hell_file, euc_file, jen_shan_file]

    for outfile in outfiles:
       if os.path.exists(outfile):
           os.remove(outfile)
    #jen_shan_file = community_dir + '/distance_info/jensen_shannon'
    #if os.path.exists(jen_shan_file):
    #    os.remove(jen_shan_file)
    
    # load the community document vector dictionary from file
    with open(community_dir + '/community_doc_vecs.pickle', 'rb') as community_doc_vecs_file:
        community_doc_vecs = pickle.load(community_doc_vecs_file)

    with open(cos_file, 'a') as cosfile, open(hell_file, 'a') as hellfile, open(euc_file, 'a') as eucfile, open(jen_shan_file, 'a') as jenshanfile:
    #with open(jen_shan_file, 'a') as jenshanfile:
        for key in sorted(community_doc_vecs):
            user = key
            # only necessary to compare each user with another user once
            vec_1 = community_doc_vecs.pop(key)

            for key_2 in sorted(community_doc_vecs):
                vec_2 = community_doc_vecs[key_2]
                cos_dist = distance.cosine(vec_1, vec_2)
                hel_dist = hellinger_distance(vec_1, vec_2)
                euc_dist = distance.euclidean(vec_1, vec_2)
                js_div = jensen_shannon_divergence(vec_1, vec_2)
                cosfile.write('{}\t{}\t{}\n'.format(user, key_2, cos_dist))
                hellfile.write('{}\t{}\t{}\n'.format(user, key_2, hel_dist))
                eucfile.write('{}\t{}\t{}\n'.format(user, key_2, euc_dist))
                jenshanfile.write('{}\t{}\t{}\n'.format(user, key_2, js_div))
    community_average_distances(community_dir)

# https://gist.github.com/larsmans/3116927
def hellinger_distance(P, Q):
    return distance.euclidean(np.sqrt(P), np.sqrt(Q)) / np.sqrt(2)

# http://stackoverflow.com/questions/15880133/jensen-shannon-distance
def jensen_shannon_divergence(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def community_average_distances(community_dir):
    #if not os.path.exists(community_dir + '/distance_info/community_average_distances'):
    tot_rows = 0
    fieldnames = ['user_1', 'user_2', 'distance']
        
    if os.path.exists(community_dir + '/distance_info/community_average_distances'):
        os.remove(community_dir + '/distance_info/community_average_distances')
    
    # load the dictionary containing the user document vectors of the community 
    with open(community_dir + '/community_doc_vecs.pickle', 'rb') as community_doc_vecs_file:
        community_doc_vecs = pickle.load(community_doc_vecs_file)
       
    # access the distance files in the community directory
    for path, dirs, files in os.walk(community_dir + '/distance_info/'):
        for distance_file in files:
            # find the average distances between users for the community
            with open(path + distance_file, 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                distances = [float(row['distance']) for row in csv_reader]
            with open(path + 'community_average_distances', 'a') as outfile:
                outfile.write('{}\t{}\n'.format(str(distance_file), scipy.mean(distances)))
    
def combine_vector_dictionaries(user_topics_dir, community_doc_vecs):
    if os.path.exists(user_topics_dir + 'all_community_doc_vecs.pickle'):
        with open(user_topics_dir + 'all_community_doc_vecs.pickle', 'rb') as all_community_file:
            all_community_doc_vecs = pickle.load(all_community_file)
    else:
        all_community_doc_vecs = {}
    all_community_doc_vecs.update(community_doc_vecs)
    with open(user_topics_dir + 'all_community_doc_vecs.pickle', 'wb') as all_community_doc_vecs_file:
        pickle.dump(all_community_doc_vecs, all_community_doc_vecs_file, -1)

# create a list of lines from pre-processed tweet input file
def convert_to_doc(tweet):
    with open(tweet, 'r') as infile:
        text = ' '.join(line.rstrip('\n') for line in infile)
        # remove emoji's and links from tweets
        # http://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
        try:
            reg_ex = re.compile(u'([\U0001F300-\U0001F64F])|([\U0001F680-\U0001F6FF])|([\U00002600-\U000027BF])')
        except:
            reg_ex = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
        text = reg_ex.sub('', text)
        # http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        text = re.sub(r'[^\w]', ' ', text) # remove hashtag
        #return list(utils.simple_preprocess(text, deacc=True, min_len=2, max_len=15))
        return list(utils.lemmatize(text))

def get_user_document_vectors(tweetpath, user_id, community_dir, user_topics_dir, community_doc_vecs):
    all_community_doc_vecs = {}
    if os.path.exists(user_topics_dir + 'all_community_doc_vecs.pickle'):
        with open(user_topics_dir + 'all_community_doc_vecs.pickle', 'rb') as all_community_file:
            all_community_doc_vecs = pickle.load(all_community_file)

    if not user_id in all_community_doc_vecs:
        document = convert_to_doc(tweetpath)

        # load wiki dictionary
        dictionary = corpora.Dictionary.load('../data/author_topic_2.dict')

        # create bag of words from input document
        doc_bow = dictionary.doc2bow(document)

        # load trained wiki model from file
        lda_model = models.LdaModel.load('../data/at_100_lem_5_pass_2.model')

        # queries the document against the LDA model and associates the data with probabalistic topics
        doc_lda = get_doc_topics(lda_model, doc_bow)
        dense_vec = gensim.matutils.sparse2full(doc_lda, lda_model.num_topics)
    
        # build dictionary of user document vectors <k, v>(user_id, vec)
        community_doc_vecs[user_id] = dense_vec
    else:
        community_doc_vecs[user_id] = all_community_doc_vecs[user_id]
    
# http://stackoverflow.com/questions/17310933/document-topical-distribution-in-gensim-lda
# mimics a dense vector representation of document bag of words input
def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])
    return [(topic_id, topic_value) for topic_id, topic_value in enumerate(topic_dist)]
        
def main():
    user_topics_dir = 'user_topics/'
    community_tweet_dirs = []
    #get_random_user_follower_vecs(user_topics_dir)

    # create output directories
    if not os.path.exists(os.path.dirname(user_topics_dir)):
        os.makedirs(os.path.dirname(user_topics_dir), 0o755)

    for path, dirs, files in os.walk('tweets_dir/'):
        for community in sorted(dirs):
            if not os.path.exists(os.path.dirname(user_topics_dir + str(community) + '/')):
                os.makedirs(os.path.dirname(user_topics_dir + str(community) + '/'), 0o755)
            if not os.path.exists(os.path.dirname(user_topics_dir + str(community) + '/distance_info/')):
                os.makedirs(os.path.dirname(user_topics_dir + str(community) + '/distance_info/'), 0o755)

            community_doc_vecs = {}
            community_dir = user_topics_dir + str(community)
            tmp_doc_vecs = []
            if os.path.exists(community_dir + '/community_doc_vecs.pickle'):
                with open(community_dir + '/community_doc_vecs.pickle', 'rb') as tmp_doc_vecs_file:
                    tmp_doc_vecs = pickle.load(tmp_doc_vecs_file)

            progress_label = 'Getting document vectors for: ' + str(community)
            with click.progressbar(os.listdir('tweets_dir/' + community), label=progress_label) as bar:
                for user_id in bar:
                    tweetpath = os.path.join('tweets_dir/' + community, user_id)
                    if tmp_doc_vecs:
                        if user_id not in tmp_doc_vecs:
                            get_user_document_vectors(tweetpath, user_id, community_dir, user_topics_dir, community_doc_vecs)
                        else:
                            community_doc_vecs[user_id] = tmp_doc_vecs[user_id]
                            continue
                    else:
                        get_user_document_vectors(tweetpath, user_id, community_dir, user_topics_dir, community_doc_vecs)

            combine_vector_dictionaries(user_topics_dir, community_doc_vecs)

            # save each community document vector dictionary for later use
            with open(community_dir + '/community_doc_vecs.pickle', 'wb') as community_doc_vecs_file:
                pickle.dump(community_doc_vecs, community_doc_vecs_file, -1)
            
            # output the user distances to file
            #delete_distance_files(community_dir)
            community_user_distances(community_dir)
        break

if __name__ == '__main__':
    sys.exit(main())
