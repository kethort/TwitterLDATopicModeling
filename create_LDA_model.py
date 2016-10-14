# Source for derivation/implementation of this code: http://radimrehurek.com/gensim/index.html
import gensim
from gensim.corpora import MmCorpus, Dictionary
import logging
import bz2
import pyLDAvis
import pyLDAvis.gensim as gensimvis
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NUM_TOPICS = 75

bow_corpus = MmCorpus('data/wiki_corpus.mm') 
dictionary = Dictionary.load('data/wiki_dict.dict')

lda = gensim.models.LdaMulticore(corpus=bow_corpus, id2word=dictionary, num_topics=NUM_TOPICS, workers=7, passes=5)
lda.save('data/lda_75_lem_5_pass.model')

vis_data = gensimvis.prepare(lda, bow_corpus, dictionary)
pyLDAvis.save_html(vis_data, 'data/lda_75_lem_5_pass.html')
