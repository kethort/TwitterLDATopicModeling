# -*- coding: utf-8 -*-
import numpy as np
import json
import sys
import re
import os
import ast
import multiprocessing
from functools import partial
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import gensim
from gensim import utils, corpora, models

ignore_words = set(stopwords.words('english'))

# http://stackoverflow.com/questions/15365046/python-removing-pos-tags-from-a-txt-file
def write_topn_words(output_dir, lda_model):
    if not os.path.exists(output_dir + 'topn_words.txt'):
        print('Writing topn words for LDA model')
        reg_ex = re.compile('(?<![\s/])/[^\s/]+(?![\S/])')

        with open(output_dir + 'topn_words.txt', 'w') as outfile:
            for i in range(lda_model.num_topics):
                outfile.write('{}\n'.format('Topic #' + str(i + 1) + ': '))
                for word, prob in lda_model.show_topic(i, topn=20):
                    word = reg_ex.sub('', word)
                    outfile.write('\t{}\n'.format(word.encode('utf-8')))
                outfile.write('\n')	

def preprocess_tweet(document):
    with open(document, 'r') as infile:
        text = ' '.join(line.rstrip('\n') for line in infile)
    # convert string into unicode
    text = gensim.utils.any2unicode(text)

    # remove URL's
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

    # remove symbols excluding the @, # and \s symbol
    text = re.sub(r'[^\w@#\s]', '', text)
    
    return utils.lemmatize(text, allowed_tags=re.compile('(NN)'), stopwords=ignore_words, min_length=3)

#    # tokenize words using NLTK Twitter Tokenizer
#    tknzr = TweetTokenizer()
#    text = tknzr.tokenize(text)
#
#    # lowercase, remove words less than len 2 & remove numbers in tokenized list
#    text = [word.lower() for word in text if len(word) > 2 and not word.isdigit()]
#
#    # remove stopwords
#    return [word for word in text if not word in ignore_words]

def get_document_vectors(user_id, **kwargs):
    print('Getting document vectors for: ' + user_id)

    if os.path.exists(kwargs['tweets_dir'] + user_id):
        tweetpath = kwargs['tweets_dir'] + user_id
    else:
        return

    if not user_id in kwargs['document_vectors']:
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

# topology: topology file, output_dir: name of directory to create, dict_loc: dictionary, lda_loc: lda model,
# dir_prefix: prefix for subdirectories (ie community_1)

# python2.7 tweets_on_LDA.py communities output_dir_name data/twitter/tweets.dict data/twitter/tweets_100_lem_5_pass.model community
def main(topology, tweets_loc, output_dir, dict_loc, lda_loc, dir_prefix):
    output_dir += '/'

    # create working folder
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir), 0o755)

    # load wiki dictionary
    model_dict = corpora.Dictionary.load(dict_loc)

    # load trained wiki model from file
    lda = models.LdaModel.load(lda_loc)

    write_topn_words(output_dir, lda)

    with open(topology, 'r') as inp_file:
        users = set(str(user) for community in inp_file for user in ast.literal_eval(community))

    try:
        with open(output_dir + 'document_vectors.json', 'r') as all_community_file:
            document_vectors = json.load(all_community_file)
    except:
        document_vectors = {}

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    func = partial(get_document_vectors, 
                   tweets_dir=tweets_loc, 
                   document_vectors=document_vectors, 
                   dictionary=model_dict, 
                   lda_model=lda) 
    doc_vecs = pool.map(func, users)
    doc_vecs = [item for item in doc_vecs if item is not None]
    pool.close()
    doc_vecs = dict(doc_vecs)

    with open(output_dir + 'document_vectors.json', 'w') as document_vectors_file:
        json.dump(doc_vecs, document_vectors_file, sort_keys=True, indent=4)

    print('Building directories')
    with open(topology, 'r') as topology_file:
        for i, community in enumerate(topology_file):
            community_dir = output_dir + dir_prefix + '_' + str(i) + '/'

            if not os.path.exists(os.path.dirname(community_dir)):
                os.makedirs(os.path.dirname(community_dir), 0o755)
            comm_doc_vecs = community_document_vectors(doc_vecs, community)
            with open(community_dir + 'community_doc_vecs.json', 'w') as comm_docs_file:
                json.dump(comm_doc_vecs, comm_docs_file, sort_keys=True, indent=4)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
