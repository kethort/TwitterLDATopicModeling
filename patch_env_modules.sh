#!/bin/bash

# patch Gensim modules and download stopword lists
printf 'd\nstopwords\nq' | python -c 'import nltk; nltk.download()'

# wikicorpus patch 
# 1. increases the min size of acceptable articles to 200 words
# 2. uses a stopword list to filter words during lemmatization 
# 3. selects only noun POS tags during lemmatization

cp patches/wikicorpus.py venv/lib/python2.7/site-packages/gensim/corpora/
cp patches/prog_class.py venv/lib/python2.7/site-packages/pyprind/prog_class.py
cp patches/english ~/nltk_data/corpora/stopwords/
