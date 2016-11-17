# Source for Gensim library and tutorials: http://radimrehurek.com/gensim/index.html
# http://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
# http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
# https://pymotw.com/2/multiprocessing/mapreduce.html
import logging
import os
import sys
import bz2
import re
import json
import itertools
import tarfile
import multiprocessing
import gensim
from gensim.corpora import MmCorpus, Dictionary, WikiCorpus
import pyLDAvis
import pyLDAvis.gensim as gensimvis

DEFAULT_DICT_SIZE = 100000

from nltk.corpus import stopwords
ignore_words = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def list_to_gen(tweets):
    for filename in tweets:
        yield filename

def preprocess_tweet(tweet):
    with open(tweet, 'r') as infile:
        # transform tweet document into one string
        text = ' '.join(line.rstrip('\n') for line in infile)
        # remove emoji's
        try:
            reg_ex = re.compile(u'([\U0001F300-\U0001F64F])|([\U0001F680-\U0001F6FF])|([\U00002600-\U000027BF])')
        except:
            reg_ex = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
        # remove URLS
        text = reg_ex.sub('', text)
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        # remove hashtag symbol
        text = re.sub(r'[^\w]', ' ', text) 
        text = text.replace("'", "")
        # remove stopwords and lemmatize
        #return list(gensim.utils.lemmatize(text, allowed_tags=re.compile('(NN)'), stopwords=ignore_words, min_length=3))
        return list(gensim.utils.lemmatize(text, stopwords=ignore_words, min_length=3))

class Document_Corpus(gensim.corpora.TextCorpus):
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

    vis_data = gensimvis.prepare(lda, bow_corpus, dictionary)
    pyLDAvis.save_html(vis_data, lda_loc + '.html')

# arg1: text or wiki corpus selector, arg2: directory of text docs or location of wiki dump
# arg3: name of output corpus, arg4: name of output dictionary,
# arg5: number of topics for model, arg6: name/location of output model

# python2.7 create_LDA_model.py t dnld_tweets/ tweet_corpus tweet_dict 100 lda_model
def main(arg1, arg2, arg3, arg4, arg5, arg6):
    if arg1 == 't':
        
        with open('inactive_users.json', 'r') as infile:
            inactive = json.load(infile)
        
        dir_list = []
        for path, dirs, files in os.walk(arg2):
            for filename in files:
                if not str(filename) in inactive:
                    dir_list.append(path + filename)
            break
        print(len(dir_list))

        doc_corpus = Document_Corpus(dir_list)

        # ignore words that appear in less than 5 documents or more than 5% of documents
        doc_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(arg3 + '.mm', doc_corpus)
        tweet_corpus.dictionary.save(arg4 + '.dict')

        build_LDA_model(arg3, arg4, arg5, arg6)

    if arg1 == 'w':
        wiki_corpus = WikiCorpus(arg2)
        wiki_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

        MmCorpus.serialize(arg3 + '.mm', wiki_corpus)
        wiki_corpus.dictionary.save(arg4 + '.dict')

        build_LDA_model(arg3, arg4, arg5, arg6)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
