import logging
import os
import sys
import bz2
import re
import itertools
import tarfile
import multiprocessing
import gensim
from gensim.corpora import MmCorpus, Dictionary, WikiCorpus
from gensim import models
from pyLDAvis import gensim as gensim_vis
import argparse
from nltk.tokenize import TweetTokenizer

DEFAULT_DICT_SIZE = 100000

from nltk.corpus import stopwords
ignore_words = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    
    # tokenize words using NLTK Twitter Tokenizer
    tknzr = TweetTokenizer()
    text = tknzr.tokenize(text)

    # lowercase & remove stopwords in tokenized list
    return [word.lower() for word in text if len(word) > 2 and word not in ignore_words]

def list_to_gen(directory):
    for filename in os.listdir(directory):
        yield directory + str(filename)

class TweetCorpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
        for tokens in pool.imap(preprocess_tweet, list_to_gen(self.input)):
            yield tokens
        pool.terminate()

def build_LDA_model(corp_loc, dict_loc, num_topics, lda_loc):
    corpus = MmCorpus(corp_loc) 
    dictionary = Dictionary.load(dict_loc)

    lda = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=int(num_topics), alpha='asymmetric', passes=5)
    lda.save(lda_loc + '.model')

    build_pyLDAvis_output(corp_loc, dict_loc, lda_loc)

def build_pyLDAvis_output(corp_loc, dict_loc, lda_loc):
    if not 'model' in lda_loc:
        lda_loc += '.model'

    corpus = MmCorpus(corp_loc)
    dictionary = Dictionary.load(dict_loc)
    lda = models.LdaModel.load(lda_loc)
    
    vis_data = gensim_vis.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis_data, lda_loc.split('.')[0] + '.html')

# option: text or wiki corpus selector, docs_loc: directory of text docs or location of wiki dump
# corp_loc: name of output corpus, output_dict: name of output dictionary,
# num_topics: number of topics for model, output_model: name/location of output model

# python2.7 create_LDA_model.py t docs_dir/ corpus dictionary 100 lda_model
def main():
    parser = argparse.ArgumentParser(description='Create a corpus from a collection of documents and/or build an LDA model')
    subparsers = parser.add_subparsers(dest='mode')
    
    text_corpus_parser = subparsers.add_parser('text', help='Build corpus from directory of text files')
    text_corpus_parser.add_argument('-d', '--docs_loc', required=True, action='store', dest='docs_loc', help='Directory where text documents stored')
    text_corpus_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location and name to save corpus')

    wiki_corpus_parser = subparsers.add_parser('wiki', help='Build corpus from compressed Wikipedia articles')
    wiki_corpus_parser.add_argument('-w', '--wiki_loc', required=True, action='store', dest='wiki_loc', help='Location of compressed Wikipedia dump')
    wiki_corpus_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location and name to save corpus')

    lda_model_parser = subparsers.add_parser('lda', help='Create LDA model from saved corpus')
    lda_model_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location of corpus')
    lda_model_parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary')
    lda_model_parser.add_argument('-n', '--num_topics', required=True, action='store', dest='num_topics', help='Number of topics to assign to LDA model')
    lda_model_parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location and name to save LDA model')

    lda_vis_parser = subparsers.add_parser('ldavis', help='Create visualization of LDA model')
    lda_vis_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location of corpus')
    lda_vis_parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary')
    lda_vis_parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location of LDA model')

    args = parser.parse_args()

    if args.mode == 'text':
        doc_corpus = TweetCorpus(args.docs_loc)

        # ignore words that appear in less than 5 documents or more than 5% of documents
        doc_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(args.corp_loc + '.mm', doc_corpus)
        doc_corpus.dictionary.save(args.corp_loc + '.dict')

    if args.mode == 'wiki':
        wiki_corpus = WikiCorpus(args.wiki_loc)
        wiki_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(args.corp_loc + '.mm', wiki_corpus)
        wiki_corpus.dictionary.save(args.corp_loc + '.dict')

    if args.mode == 'lda':
        build_LDA_model(args.corp_loc, args.dict_loc, args.num_topics, args.lda_loc)

    if args.mode == 'ldavis':
        build_pyLDAvis_output(args.corp_loc, args.dict_loc, args.lda_loc)

if __name__ == '__main__':
    sys.exit(main())
