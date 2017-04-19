# -*- coding: utf-8 -*-
import numpy as np
import time
import gensim
from gensim import utils, corpora, models
import json
import sys
import re
import os
import ast
import click
import subprocess
import multiprocessing
from functools import partial
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

ignore_words = set(stopwords.words('english'))

# http://stackoverflow.com/questions/15365046/python-removing-pos-tags-from-a-txt-file
def write_topn_words(user_topics_dir, lda_model):
    if not os.path.exists(user_topics_dir + 'topn_words.txt'):
        print('Writing topn words for LDA model')
        reg_ex = re.compile('(?<![\s/])/[^\s/]+(?![\S/])')

        with open(user_topics_dir + 'topn_words.txt', 'w') as outfile:
            for i in range(lda_model.num_topics):
                outfile.write('{}\n'.format('Topic #' + str(i + 1) + ': '))
                for word, prob in lda_model.show_topic(i, topn=20):
                    word = reg_ex.sub('', word)
                    outfile.write('\t{}\n'.format(word.encode('utf-8')))
                outfile.write('\n')	

def combine_vector_dictionaries(user_topics_dir, community_doc_vecs):
    try:
        with open(user_topics_dir + 'all_community_doc_vecs.json', 'r') as all_community_file:
            all_community_doc_vecs = json.load(all_community_file)
    except:
        all_community_doc_vecs = {}

    all_community_doc_vecs.update(community_doc_vecs)
    with open(user_topics_dir + 'all_community_doc_vecs.json', 'w') as all_community_doc_vecs_file:
        json.dump(all_community_doc_vecs, all_community_doc_vecs_file, sort_keys=True, indent=4)

def preprocess_tweet(document):
    with open(document, 'r') as infile:
        # transform document into one string
        text = ' '.join(line.rstrip('\n') for line in infile)
    # convert string into unicode
    text = gensim.utils.any2unicode(text)

    # remove URL's
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

    # remove symbols excluding the @, # and \s symbol
    text = re.sub(r'[^\w@#\s]', '', text)
    
	#return utils.lemmatize(text)

    # tokenize words using NLTK Twitter Tokenizer
    tknzr = TweetTokenizer()
    text = tknzr.tokenize(text)

    # lowercase, remove words less than len 2 & remove numbers in tokenized list
    text = [word.lower() for word in text if len(word) > 2 and not word.isdigit()]

    # remove stopwords
    return [word for word in text if not word in ignore_words]

def get_document_vectors(user_id, **kwargs):
    if 'clique' in user_id:
        print('Getting document vectors for: ' + user_id)

    if os.path.exists(kwargs['tweets_dir'] + user_id):
        tweetpath = kwargs['tweets_dir'] + user_id
    else:
        return

    if not user_id in kwargs['all_comm_doc_vecs']:
        document = preprocess_tweet(tweetpath)

        # if after preprocessing the list is empty then skip that user
        if not document:
            return

        # create bag of words from input document
        doc_bow = kwargs['dictionary'].doc2bow(document)

        # queries the document against the LDA model and associates the data with probabalistic topics
        doc_lda = get_doc_topics(kwargs['lda_model'], doc_bow)
        dense_vec = gensim.matutils.sparse2full(doc_lda, kwargs['lda_model'].num_topics)
    
        # build dictionary of user document vectors <k, v>(user_id, vec)
        return (user_id, dense_vec.tolist())
    else:
        return (user_id, kwargs['all_comm_doc_vecs'][user_id])
    
# http://stackoverflow.com/questions/17310933/document-topical-distribution-in-gensim-lda
def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])
    return [(topic_id, topic_value) for topic_id, topic_value in enumerate(topic_dist)]

def users_to_iter(community):
    for user in ast.literal_eval(community):
        yield str(user)

# topology: topology file, output_dir: name of directory to create, dict_loc: dictionary, lda_loc: lda model,
# dir_prefix: prefix for subdirectories (ie community_1)

# python2.7 tweets_on_LDA.py communities user_topics_ex data/twitter/tweets.dict data/twitter/tweets_100_lem_5_pass.model community
def main(topology, tweets_loc, output_dir, dict_loc, lda_loc, dir_prefix):
    user_topics_dir = output_dir + '/'

    # create output directories
    if not os.path.exists(os.path.dirname(user_topics_dir)):
        os.makedirs(os.path.dirname(user_topics_dir), 0o755)

    # load wiki dictionary
    model_dict = corpora.Dictionary.load(dict_loc)

    # load trained wiki model from file
    lda = models.LdaModel.load(lda_loc)

    write_topn_words(user_topics_dir, lda)

    with open(topology, 'r') as topology_file:
        for i, community in enumerate(topology_file):
            try:
                with open(user_topics_dir + 'all_community_doc_vecs.json', 'r') as all_community_file:
                    all_community_doc_vecs = json.load(all_community_file)
            except:
                all_community_doc_vecs = {}

            community_dir = user_topics_dir + dir_prefix + '_' + str(i) + '/'
 
            if not os.path.exists(os.path.dirname(community_dir)):
                os.makedirs(os.path.dirname(community_dir), 0o755)

            print('Getting document vectors for %s %s ' % (dir_prefix, i))
             
            community_doc_vecs = {}
            pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
            func = partial(get_document_vectors, 
                           tweets_dir=tweets_loc, 
                           all_comm_doc_vecs=all_community_doc_vecs, 
                           dictionary=model_dict, 
                           lda_model=lda)
            doc_vecs = pool.map(func, users_to_iter(community))
            doc_vecs = [item for item in doc_vecs if item is not None]
            pool.close()
            community_doc_vecs = dict(doc_vecs)
 
            combine_vector_dictionaries(user_topics_dir, community_doc_vecs)
 
            # save each community document vector dictionary for later use
            with open(community_dir + '/community_doc_vecs.json', 'w') as community_doc_vecs_file:
                json.dump(community_doc_vecs, community_doc_vecs_file, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
