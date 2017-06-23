# twitter_LDA_topic_modeling

Using supervised learning techniques, it's possible to classify a collection of documents into latent topics, providing a mechanism for 
determining similarities between other documents. Twitter users associate and socialize with other users based on similar 
interests. The Tweets of these users can be classified using a trained LDA model to automate the discovery of their similarities. 

### Prerequisites

Python 2.7 is required for use because Pattern package not compatible with Python > 2.7.



### Installing

Download:

```
git clone https://github.com/kenneth-orton/twitter_LDA_topic_modeling.git
```

Run linux_setup.sh:

```
./linux_setup.sh
```

The requirements.txt file contains the necessary Python packages.

Install using: 

```
pip2 -r install requirements.txt
```

### Process

1. Download Tweets for each user - get_community_tweets.py
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
