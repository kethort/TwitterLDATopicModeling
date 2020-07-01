# Description
Twitter users often associate and socialize with other users based on similar interests. The Tweets of these users can be classified using a trained LDA model to automate the discovery of their similarities. 

### Prerequisites

To use Python 3, the beta version of the Pattern library must be manually installed. As of the last commit of this repository, the beta version of Pattern is still incompatible with Python 3.

```
pip3 install git+git://github.com/pattern3/pattern.git
```

Python 2.7 is recommended.

If you manually install Pattern3 you should remove the pattern library from the requirements.txt file 
before installing.

### Installing

Download:

```
git clone https://github.com/kethort/twitter_LDA_topic_modeling.git
```

Run linux_setup.sh:

```
./linux_setup.sh
```

Install Python packages using pip (or use an environment like a normal person): 

```
# for Python 2.7
pip install -r requirements2.txt

# for Python 3
pip install -r requirements3.txt
```

Link to the simple-wikipedia dump:

```
https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
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
