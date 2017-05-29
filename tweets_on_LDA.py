# -*- coding: utf-8 -*-
import numpy as np
import json
import sys
import re
import os
import ast
import argparse
import argcomplete
import multiprocessing
from functools import partial
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import gensim
from gensim import utils, corpora, models
ignore_words = set(stopwords.words('english'))

def write_topn_words(output_dir, lda):
    if not os.path.exists(output_dir + 'topn_words.json'):
        print('Writing topn words for LDA model')
        reg_ex = re.compile('(?<![\s/])/[^\s/]+(?![\S/])')
        topn_words = {'Topic ' + str(i + 1): [reg_ex.sub('', word) for word, prob in lda.show_topic(i, topn=20)] for i in range(0, lda.num_topics)}
        with open(output_dir + 'topn_words.json', 'w') as outfile:
            json.dump(topn_words, outfile, sort_keys=True, indent=4)

def preprocess_tweet(document, lemma):
    with open(document, 'r') as infile:
        text = ' '.join(line.rstrip('\n') for line in infile)
    # convert string into unicode
    text = gensim.utils.any2unicode(text)
    # remove URL's
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    # remove symbols excluding the @, # and \s symbol
    text = re.sub(r'[^\w@#\s]', '', text)
    if lemma:
        return utils.lemmatize(text, stopwords=ignore_words, min_length=3)
    # tokenize words using NLTK Twitter Tokenizer
    tknzr = TweetTokenizer()
    text = tknzr.tokenize(text)
    # lowercase, remove words less than len 2 & remove numbers in tokenized list
    text = [word.lower() for word in text if len(word) > 2 and not word.isdigit()]
    # remove stopwords
    return [word for word in text if not word in ignore_words]

def get_document_vectors(user_id, **kwargs):
    print('Getting document vectors for: ' + user_id)
    if os.path.exists(kwargs['tweets_dir'] + user_id):
        tweetpath = kwargs['tweets_dir'] + user_id
    else:
        return
    if not user_id in kwargs['document_vectors']:
        document = preprocess_tweet(tweetpath, kwargs['lemma'])
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
        return (user_id, kwargs['document_vectors'][user_id])
    
# http://stackoverflow.com/questions/17310933/document-topical-distribution-in-gensim-lda
def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])
    return [(topic_id, topic_value) for topic_id, topic_value in enumerate(topic_dist)]

def community_document_vectors(doc_vecs, community):
    comm_doc_vecs = {}
    for user in ast.literal_eval(community):
        try:
            comm_doc_vecs[str(user)] = doc_vecs[str(user)]
        except:
            pass
    return comm_doc_vecs

def main():
    parser = argparse.ArgumentParser(description='Create a corpus from a collection of tweets and/or build an LDA model')
    parser.add_argument('-t', '--topology_file', required=True, action='store', dest='top_file', help='Location of topology file')
    parser.add_argument('-p', '--dir_prefix', choices=['clique', 'community'], required=True, action='store', dest='dir_prefix', help='Select whether the topology contains cliques or communities')
    parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='Name of the directory you want to direct output to')
    parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location of the saved LDA model')
    parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary for the model')
    parser.add_argument('-m', '--lemma', action='store_true', dest='lemma', help='Use this option to lemmatize words')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    output_dir = args.working_dir + '/'
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), 0o755)

    # load dictionary
    model_dict = corpora.Dictionary.load(args.dict_loc)
    # load trained model from file
    lda = models.LdaModel.load(args.lda_loc)
    write_topn_words(output_dir, lda)

    with open(args.top_file, 'r') as inp_file:
        users = set(str(user) for community in inp_file for user in ast.literal_eval(community))
    try:
        with open(output_dir + 'document_vectors.json', 'r') as all_community_file:
            document_vectors = json.load(all_community_file)
    except:
        document_vectors = {}

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    func = partial(get_document_vectors, 
                   tweets_dir='dnld_tweets/', 
                   document_vectors=document_vectors, 
                   dictionary=model_dict, 
                   lda_model=lda,
                   lemma=args.lemma) 
    doc_vecs = pool.map(func, users)
    doc_vecs = [item for item in doc_vecs if item is not None]
    pool.close()
    doc_vecs = dict(doc_vecs)

    with open(output_dir + 'document_vectors.json', 'w') as document_vectors_file:
        json.dump(doc_vecs, document_vectors_file, sort_keys=True, indent=4)

    print('Building directories')
    with open(args.top_file, 'r') as topology_file:
        for i, community in enumerate(topology_file):
            community_dir = output_dir + args.dir_prefix + '_' + str(i) + '/'
            if not os.path.exists(os.path.dirname(community_dir)):
                os.makedirs(os.path.dirname(community_dir), 0o755)
            comm_doc_vecs = community_document_vectors(doc_vecs, community)
            with open(community_dir + 'community_doc_vecs.json', 'w') as comm_docs_file:
                json.dump(comm_doc_vecs, comm_docs_file, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main())
