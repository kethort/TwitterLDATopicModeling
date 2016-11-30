import copy
import gensim
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import ast

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

# https://stackoverflow.com/questions/13249415/can-i-implement-custom-indentation-for-pretty-printing-in-python-s-json-module
class MyJSONEncoder(json.JSONEncoder):
  def iterencode(self, o, _one_shot=False):
    list_lvl = 0
    for s in super(MyJSONEncoder, self).iterencode(o, _one_shot=_one_shot):
      if s.startswith('['):
        list_lvl += 1
        s = s.replace('\n', '').rstrip()
      elif 0 < list_lvl:
        s = s.replace('\n', '')
        if s and s[-1] == ',':
          s = s[:-1] + self.item_separator
        elif s and s[-1] == ':':
          s = s[:-1] + self.key_separator
      if s.endswith(']'):
        list_lvl -= 1
      yield s

def serialize_json(json_filepath):
    try:
    	with open(json_filepath, 'r') as infile:
    	    all_docs = json.load(infile)
    except:
    	all_docs = {}
    
    if all_docs:
        with open(json_filepath, 'w') as outfile:
        json.dump(all_docs, outfile, cls=MyJSONEncoder, sort_keys=True, indent=2)
