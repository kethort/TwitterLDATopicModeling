# Source for Gensim library and tutorials: http://radimrehurek.com/gensim/index.html
# http://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
# http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
# https://pymotw.com/2/multiprocessing/mapreduce.html
import logging
import os
import sys
import re
import itertools
import tarfile
import multiprocessing
import gensim
from gensim.corpora import MmCorpus, Dictionary

DEFAULT_DICT_SIZE = 100000

from nltk.corpus import stopwords
ignore_words = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def list_to_gen(tweets):
    for filename in tweets:
        yield 'tweets_dir/' + str(filename)

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
        return list(gensim.utils.lemmatize(text, allowed_tags=re.compile('(NN)'), stopwords=ignore_words, min_length=3)

class Text_Corpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        pool = multiprocessing.Pool(4)
        for tokens in pool.imap(preprocess_tweet, self.input):
            yield tokens
        pool.terminate()

tweets = list_to_gen(os.listdir('tweets_dir/'))
tweet_corpus = Text_Corpus(tweets)
# ignore words that appear in less than 5 documents or more than 5% of documents
tweet_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

MmCorpus.serialize('data/sample_tweet_corpus.mm', tweet_corpus, progress_cnt=2000)
tweet_corpus.dictionary.save('data/sample_tweet_dict.dict')

