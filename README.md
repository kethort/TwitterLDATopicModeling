# twitter_LDA_topic_modeling
Uses the Gensim library in Python to create a corpus from documents and train an LDA model.
The model is used to evaluate similarities in conversations between Twitter users.
The groups of Twitter users being evaluated were found using community detection algorithms. Tweepy was used to communicate with the Twitter API. Matplotlib and pyLDAvis were used to produce illustrations of the data. Scipy, Numpy and NLTK were used during processing of the data. 

## Premise
The follower relationship on Twitter differs from other social media networks like Facebook because users who create friendships on Twitter don't always do so based on prior human interaction. Many Twitter users follow others based purely on shared interests. Using the Twitter API, it's possible to isolate sections of localized populations who have formed communities based on a 1:1 follower relationship between other users. These communities of users often display a central purpose or associative theme of shared interests. The goal of this project is to identify topics in the discussions of Twitter users who are associated in online communities. If a topical model can categorize discussions between users of these communities, it may be possible to rank the users in the communities by activity and users that don't share the communities interests can be identified. 

## Description of Data
This project uses two topologies generated in 2014 by another student using Python, NetworkX and Tweepy to locate Twitter users within a geo-bounded area. Each line in the topology is a list of users who share a 1:1 follower relationship with the other users in the list. 

<p align="center">
  <img src="/img/topology.png" width="500"/>
</p>

One topology represents condensed communities which will be referred to as a clique, and the other topology file contains larger cliques which will be referred to as a community. The topologies being used contain 1897 cliques and the same number of communities. Users can belong to more than one clique or community, and the total amount of unique users in the entire data set is 24,838. The Twitter REST API allows a maximum of 3,200 of the most recent Tweets to be downloaded from each user's timeline. The average amount of Tweets downloaded for each unique user in the data set is 1462. The following graph shows the distribution of the total amount of Tweets across all users' timelines. 

<p align="center">
  <img src="/img/tweet_distribution.png" width="500"/>
</p>

Users who have less than 5 Tweets on their timeline are considered inactive and are omitted from the data set. Users who have restricted access to their timeline are regarded as not having any Tweets on their timeline. Omitting these users removes 1286 users from the data set leaving 23,552 unique users. The resulting minimum clique size is 2, the average clique size is 5 and the maximum clique size is 36. The resulting minimum community size is 3, the average community size is 30 and the maximum community size is 318. The following graphs represent the distribution of clique and community sizes. 

<p align="center">
  <img src="/img/clique_size_distribution.png" width="350"/><img src="/img/community_size_distribution.png" width="350"/>
</p>

## Approach
