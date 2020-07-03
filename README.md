# Description
Twitter users often associate and socialize with other users based on similar interests. The Tweets of these users can be classified using a trained LDA model to automate the discovery of their similarities. 

### Prerequisites

Python 2.7 is recommended since the pattern library is currently incompatible with most Python 3 versions.

Python 3.6 can be used with the pattern library, though it may need to be built from source since most newer Linux distributions don't come with it pre-installed. The commands to build Python 3.6 from source are provided in the linux_setup_py3.6.sh script.


### Installing

## Linux

Download:

```
git clone https://github.com/kethort/twitter_LDA_topic_modeling.git
```

Run bash script:

```
./linux_setup_py3.6.sh
```

Python pip requirements included in these files:

```
# for Python 2.7
pip install -r requirements_py2.txt

# for Python 3
pip install -r requirements_py3.txt
```

Link to the simple-wikipedia dump:

```
https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
```
## Mac osx
The installation is very similar to the linux installation:

```
extra install instructions in osx_setup_py3.6.info

pip install -r requirements_py3_OSX.txt
```

### Process

1. Get user and follower ids by location - twitter_user_grabber.py
2. Download Tweets for each user - get_community_tweets.py
2. Create an LDA model from a corpus of documents - create_LDA_model.py
3. Generate topic probability distributions for Tweet documents - tweets_on_LDA.py
4. Calculate distances between Tweet documents and graph them - plot_distances.py

### Sample Visualizations

<p align="center">
  <img src="/img/user_x_distribution.png" width="500"/>
</p>

<p align="center">
  <img src="/img/user_x_lda_vis.png"/>
</p>

<p align="center">
  <img src="/img/user_internal_external.png" width="500"/>
</p>

<p align="center">
  <img src="/img/community_median_internal_external.png" width="500"/>
</p>

## Built With

* [Gensim](https://radimrehurek.com/gensim/) - Package for creating LDA model
* [pyLDAvis](https://github.com/bmabey/pyLDAvis) - Package for visualizing LDA model
* [Tweepy](http://www.tweepy.org/) - Package for interacting with Twitter REST API
* [NLTK](http://www.nltk.org/) - Package for stopword management and tokenization
