# Description
Twitter users often associate and socialize with other users based on similar interests. The Tweets of these users can be classified using a trained LDA model to automate the discovery of their similarities. 

### Prerequisites

To use Python 3, the beta version of the Pattern library must be manually installed using:

pip install git+git://github.com/pattern3/pattern.git

Otherwise, Python 2.7 can be used since Pattern package is not currently compatible with Python > 2.7.

If you manually install Pattern3 you should remove the pattern library from the requirements.txt file 
before installing.

### Installing

Download:

```
git clone https://github.com/kenneth-orton/twitter_LDA_topic_modeling.git
```

Run linux_setup.sh:

```
./linux_setup.sh
```

Install Python packages using pip (or use an environment like a normal person): 

```
pip install -r requirements.txt
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
