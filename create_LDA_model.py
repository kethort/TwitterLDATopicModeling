import logging
import os
import sys
import bz2
import re
import itertools
import tarfile
import multiprocessing
from functools import partial
import gensim
from gensim.corpora import MmCorpus, Dictionary, WikiCorpus
from gensim import models, utils
import pyLDAvis
from pyLDAvis import gensim as gensim_vis
import argparse
import argcomplete
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

DEFAULT_DICT_SIZE = 100000
ignore_words = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def wiki_tokenizer(content, token_min_len=3, token_max_len=15, lower=True):
    return [
        utils.to_unicode(token) for token in utils.simple_preprocess(content, deacc=True, min_len=3) 
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_') and not token.isdigit()
        and not token in ignore_words
    ]

def preprocess_text(lemma, document):
    with open(document, 'r') as infile:
        # transform document into one string
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
    text = [word.lower() for word in text if len(word) > 2 and not word.isdigit() and not word in ignore_words]
    return utils.simple_preprocess(text, deacc=True, min_len=3)

def list_to_gen(directory):
    for filename in os.listdir(directory):
        yield directory + str(filename)

class DocCorpus(gensim.corpora.TextCorpus):
    def __init__(self, docs_loc, lemmatize, dictionary=None, metadata=None):
        self.docs_loc = docs_loc
        self.lemmatize = lemmatize
        self.metadata = metadata
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary
    def get_texts(self):
        pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
        func = partial(preprocess_text, self.lemmatize)
        for tokens in pool.map(func, list_to_gen(self.docs_loc)):
            yield tokens
        pool.terminate()

def build_LDA_model(corp_loc, dict_loc, num_topics, num_pass, lda_loc):
    corpus = MmCorpus(corp_loc) 
    dictionary = Dictionary.load(dict_loc)

    lda = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=int(num_topics), alpha='asymmetric', passes=int(num_pass))
    lda.save(lda_loc + '.model')

    build_pyLDAvis_output(corp_loc, dict_loc, lda_loc)

def build_pyLDAvis_output(corp_loc, dict_loc, lda_loc):
    if not 'model' in lda_loc:
        lda_loc += '.model'

    corpus = MmCorpus(corp_loc)
    dictionary = Dictionary.load(dict_loc)
    lda = models.LdaModel.load(lda_loc)
    
    vis_data = gensim_vis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis_data, lda_loc.split('.')[0] + '.html')

def main():
    parser = argparse.ArgumentParser(description='Create a corpus from a collection of tweets and/or build an LDA model')
    subparsers = parser.add_subparsers(dest='mode')
    
    text_corpus_parser = subparsers.add_parser('text', help='Build corpus from directory of text files')
    text_corpus_parser.add_argument('-d', '--docs_loc', required=True, action='store', dest='docs_loc', help='Directory where tweet documents stored')
    text_corpus_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location and name to save corpus')
    text_corpus_parser.add_argument('-m', '--lemma', action='store_true', dest='lemma', help='Use this option to lemmatize words')

    wiki_corpus_parser = subparsers.add_parser('wiki', help='Build corpus from compressed Wikipedia articles')
    wiki_corpus_parser.add_argument('-w', '--wiki_loc', required=True, action='store', dest='wiki_loc', help='Location of compressed Wikipedia dump')
    wiki_corpus_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location and name to save corpus')
    wiki_corpus_parser.add_argument('-m', '--lemma', action='store_true', dest='lemma', help='Use this option to lemmatize words')

    lda_model_parser = subparsers.add_parser('lda', help='Create LDA model from saved corpus')
    lda_model_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location of corpus')
    lda_model_parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary')
    lda_model_parser.add_argument('-n', '--num_topics', required=True, action='store', dest='num_topics', help='Number of topics to assign to LDA model')
    lda_model_parser.add_argument('-p', '--num_pass', required=True, action='store', dest='num_pass', help='Number of passes through corpus when training the LDA model')
    lda_model_parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location and name to save LDA model')

    lda_vis_parser = subparsers.add_parser('ldavis', help='Create visualization of LDA model')
    lda_vis_parser.add_argument('-c', '--corp_loc', required=True, action='store', dest='corp_loc', help='Location of corpus')
    lda_vis_parser.add_argument('-d', '--dict_loc', required=True, action='store', dest='dict_loc', help='Location of dictionary')
    lda_vis_parser.add_argument('-l', '--lda_loc', required=True, action='store', dest='lda_loc', help='Location of LDA model')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == 'text':
        doc_corpus = DocCorpus(args.docs_loc, args.lemma)

        doc_corpus.dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(args.corp_loc + '.mm', doc_corpus)
        doc_corpus.dictionary.save(args.corp_loc + '.dict')

    if args.mode == 'wiki':
        if args.lemma:
            wiki_corpus = WikiCorpus(args.wiki_loc, lemmatize=True, tokenizer_func=wiki_tokenizer, article_min_tokens=100, token_min_len=3, token_max_len=15)
        else:
            wiki_corpus = WikiCorpus(args.wiki_loc, lemmatize=False, tokenizer_func=wiki_tokenizer, article_min_tokens=100, token_min_len=3, token_max_len=15)

        wiki_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(args.corp_loc + '.mm', wiki_corpus)
        wiki_corpus.dictionary.save(args.corp_loc + '.dict')

    if args.mode == 'lda':
        build_LDA_model(args.corp_loc, args.dict_loc, args.num_topics, args.num_pass, args.lda_loc)

    if args.mode == 'ldavis':
        build_pyLDAvis_output(args.corp_loc, args.dict_loc, args.lda_loc)

if __name__ == '__main__':
    sys.exit(main())
