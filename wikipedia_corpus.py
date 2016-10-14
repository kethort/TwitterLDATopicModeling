# Source for derivation/implementation of this code: http://radimrehurek.com/gensim/index.html
DEFAULT_DICT_SIZE = 170000
import gensim
from gensim.corpora import WikiCorpus, MmCorpus, Dictionary
import bz2
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if not os.path.exists(os.path.dirname('data/')):
    os.makedirs(os.path.dirname('data/'), 0o755)

# this will take many hours! Output is Wikipedia bucket-of-words sparse matrix
articles = 'enwiki-latest-pages-articles.xml.bz2'
wiki_corpus = WikiCorpus(articles)
wiki_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

# File will be several GBs
MmCorpus.serialize('data/wiki_corpus.mm', wiki_corpus, progress_cnt=10000)
wiki_corpus.dictionary.save('data/wiki_dict.dict')
