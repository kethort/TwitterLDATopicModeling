# Source for this code: http://radimrehurek.com/gensim/index.html
import logging
import os
import sys
import re
import itertools
import tarfile
import gensim
from gensim.corpora import MmCorpus, Dictionary

DEFAULT_DICT_SIZE = 100000

from nltk.corpus import stopwords
ignore_words = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_tweet(tweet):
    """
    Preprocess a single tweet, returning the result as
    a unicode string.
    """
    # transform tweet document into one string
    text = ' '.join(line.rstrip('\n') for line in tweet)
    # http://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python
    # remove emoji's and links from tweets
    try:
        reg_ex = re.compile(u'([\U0001F300-\U0001F64F])|([\U0001F680-\U0001F6FF])|([\U00002600-\U000027BF])')
    except:
        reg_ex = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    text = reg_ex.sub('', text)
    # http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'[^\w]', ' ', text) # remove hashtag
    text = text.replace("'", "")
    # remove stopwords
    words = [word for word in text.lower().split() if word not in ignore_words and len(word) > 2]
    #return list(utils.simple_preprocess(text, deacc=True, min_len=2, max_len=15))
    text = ' '.join(words)
    #return text
    return list(gensim.utils.lemmatize(text))

class Text_Corpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        for filename in self.input:
            with open('working_folder/tweets_dir/' + filename, 'r') as infile:
                yield preprocess_tweet(infile)

tweets = os.listdir('working_folder/tweets_dir/')
tweet_corpus = Text_Corpus(tweets)
tweet_corpus.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=DEFAULT_DICT_SIZE)

MmCorpus.serialize('data/author_topic_2.mm', author_topic_corpus, progress_cnt=2000)
tweet_corpus.dictionary.save('data/author_topic_2.dict')


