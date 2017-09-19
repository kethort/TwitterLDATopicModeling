import os, sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing
from functools import partial
import gensim
from gensim import models, matutils
from gensim.corpora import MmCorpus, Dictionary
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svdvals
from nltk.corpus import stopwords
from tqdm import tqdm
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ignore_words = frozenset(stopwords.words('english'))

def extract_data(topic_model, corpus, dictionary, doc_topic_dists=None):
    '''
    extract_data method is copied directly from gensim.py in the pyLDAvis library
    '''
    if not matutils.ismatrix(corpus):
          corpus_csc = matutils.corpus2csc(corpus, num_terms=len(dictionary))
    else:
          corpus_csc = corpus
          # Need corpus to be a streaming gensim list corpus for len and inference functions below:
          corpus = matutils.Sparse2Corpus(corpus_csc)

    # TODO: add the hyperparam to smooth it out? no beta in online LDA impl.. hmm..
    # for now, I'll just make sure we don't ever get zeros...
    fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
    doc_lengths = corpus_csc.sum(axis=0).A.ravel()

    assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(corpus))

    if hasattr(topic_model, 'lda_alpha'):
           num_topics = len(topic_model.lda_alpha)
    else:
           num_topics = topic_model.num_topics

    if doc_topic_dists is None:
          # If its an HDP model.
          if hasattr(topic_model, 'lda_beta'):
              gamma = topic_model.inference(corpus)
          else:
              gamma, _ = topic_model.inference(corpus)
          doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
    else:
          if isinstance(doc_topic_dists, list):
             doc_topic_dists = matutils.corpus2dense(doc_topic_dists, num_topics).T
          elif issparse(doc_topic_dists):
             doc_topic_dists = doc_topic_dists.T.todense()
          doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)

    assert doc_topic_dists.shape[1] == num_topics, 'Document topics and number of topics do not match {} != {}'.format(doc_topic_dists.shape[1], num_topics)

    # get the topic-term distribution straight from gensim without iterating over tuples
    if hasattr(topic_model, 'lda_beta'):
           topic = topic_model.lda_beta
    else:
           topic = topic_model.state.get_lambda()
    topic = topic / topic.sum(axis=1)[:, None]
    topic_term_dists = topic[:, fnames_argsort]

    assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]

    coherence_model = models.CoherenceModel(model=topic_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')

    return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists, 
            'doc_lengths': doc_lengths, 'u_mass_coherence': coherence_model.get_coherence(), 'num_topics': num_topics}

def cao_juan_2009(topic_term_dists, num_topics):
    cos_pdists = squareform(pdist(topic_term_dists, metric='cosine')) 
    return np.sum(cos_pdists) / (num_topics*(num_topics - 1)/2)

def arun_2010(topic_term_dists, doc_topic_dists, doc_lengths, num_topics):
    P = svdvals(topic_term_dists)
    Q = np.matmul(doc_lengths, doc_topic_dists) / np.linalg.norm(doc_lengths)
    return entropy(P, Q)

def deveaud_2014(topic_term_dists, num_topics):
    jsd_pdists = squareform(pdist(topic_term_dists, metric=jensen_shannon)) 
    return np.sum(jsd_pdists) / (num_topics*(num_topics - 1))

def jensen_shannon(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))

def preprocess_text(text):
    with open(text, 'r') as inp:
        text = ' '.join(line.rstrip('\n') for line in inp)
    return [word for word in gensim.utils.simple_preprocess(text, deacc=True, min_len=3) if word not in ignore_words]

def files_to_gen(directory):
    for path, dirs, files in os.walk(directory):
        for name in files:
            yield os.path.join(path, name)

class DocCorpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
        for tokens in pool.map(preprocess_text, files_to_gen(self.input)):
            yield tokens
        pool.terminate()

def build_coherence_models(topic_model, **kwargs):
    u_mass = models.CoherenceModel(model=topic_model, corpus=kwargs['corpus'], dictionary=kwargs['dictionary'], coherence='u_mass')
    c_v = models.CoherenceModel(model=topic_model, texts=kwargs['texts'], corpus=kwargs['corpus'], dictionary=kwargs['dictionary'], coherence='c_v')
    c_uci = models.CoherenceModel(model=topic_model, texts=kwargs['texts'], corpus=kwargs['corpus'], dictionary=kwargs['dictionary'], coherence='c_uci')
    c_npmi = models.CoherenceModel(model=topic_model, texts=kwargs['texts'], corpus=kwargs['corpus'], dictionary=kwargs['dictionary'], coherence='c_npmi')
    return {'num_topics': topic_model.num_topics, 'u_mass': u_mass.get_coherence(), 'c_v': c_v.get_coherence(), 'c_uci': c_uci.get_coherence(), 'c_npmi': c_npmi.get_coherence()}

def main(text_dir):
    topics = range(10, 101, 10) + range(120, 201, 20) + range(250, 451, 50)
    #topics = range(10, 21, 10)
    #corpus = DocCorpus(text_dir)
    #dictionary = corpus.dictionary
    corpus = MmCorpus('../twitter_LDA_topic_modeling/simple-wiki.mm')
    dictionary = Dictionary.load('../twitter_LDA_topic_modeling/simple-wiki.dict')
    print('Building LDA models')
    lda_models = [models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=i, passes=5) for i in tqdm(topics)]
    print('Generating coherence models')
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    func = partial(build_coherence_models, 
                   corpus=corpus,
                   dictionary=dictionary,
                   texts=texts)
    coherence_models = pool.map(func, lda_models)
    pool.close()
#    print('Extracting data from models')
#    model_data = [extract_data(model, corpus, dictionary) for model in tqdm(lda_models)]
#    d = defaultdict(list)
#    print('Generating output data')
#    for i, data in tqdm(enumerate(model_data)):
#        d['num_topics'].append(data['num_topics'])
#        d['cao_juan_2009'].append(cao_juan_2009(data['topic_term_dists'], data['num_topics']))
#        d['arun_2010'].append(arun_2010(data['topic_term_dists'], data['doc_topic_dists'], data['doc_lengths'], data['num_topics']))
#        d['deveaud_2014'].append(deveaud_2014(data['topic_term_dists'], data['num_topics']))
#        d['u_mass_coherence'].append(data['u_mass_coherence'])
    d = defaultdict(list)
    print('Generating output data')
    for data in tqdm(coherence_models):
        d['num_topics'].append(data['num_topics'])
        d['u_mass'].append(data['u_mass'])
        d['c_v'].append(data['c_v'])
        d['c_uci'].append(data['c_uci'])
        d['c_npmi'].append(data['c_npmi'])
    df = pd.DataFrame(d)
    df = df.set_index('num_topics')
    df.to_csv('coherence_simple_wiki', sep='\t')
    df.plot(xticks=df.index, style=['bs-', 'yo-', 'r^-', 'gx-'])
    ax1 = df.plot(xticks=df.index, style='bs-', grid=True, y='u_mass')
    ax2 = df.plot(xticks=df.index, style='yo-', grid=True, y='c_v', ax=ax1)
    ax3 = df.plot(xticks=df.index, style='r^-', grid=True, y='c_npmi', ax=ax2)
    df.plot(xticks=df.index, style='gx-', grid=True, y='c_uci', ax=ax3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fancybox=True, shadow=True, ncol=4, fontsize=9)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(df.index, rotation=45, ha='right', fontsize=8)
    plt.savefig('coherence_simple_wiki')
    plt.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
