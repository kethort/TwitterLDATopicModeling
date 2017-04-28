# twitter_LDA_topic_modeling
## Introduction
Users on social media platforms like Twitter create reciprocal friendships with each other, often forming groups of interconnected follower relationships. In the context of network theory, these relationships between groups of connected users are referred to as communities.
		There exist many different algorithms for discovering communities but little research has been done to understand the significance of the context of association within these communities. A simple approach to discovering the substance of user relationships on Twitter would be to read through the tweets of each user in the community and come to a conclusion about the interests of each user based on the observations. Unfortunately, this approach will not scale well for large amounts of data and the categorization of interests would be biased in regards to the person who is interpreting the information.
		In this project, we examine whether or not users in communities discovered by a topology-based community detection algorithm display similar interests. We use a topology of communities already discovered using an algorithm called Clique Augmentation Algorithm and we generate an LDA model using Wikipedia as a training set to identify latent topics in the tweets of each user in the communities. 


## Description of Data
This project uses two topologies generated using NetworkX and Tweepy to locate Twitter users within a geo-bounded area. Each line in the topology is a list of users who share a 1:1 follower relationship with the other users in the list. 

<p align="center">
  <img src="/img/topology.png" width="500"/>
</p>

One topology represents condensed communities which will be referred to as a clique, and the other topology file contains larger cliques which will be referred to as a community. The topologies being used contain 1897 cliques and the same number of communities. Users can belong to more than one clique or community, and the total amount of unique users in the entire data set is 24,838. The Twitter REST API allows a maximum of 3,200 of the most recent Tweets to be downloaded from each user's timeline. The average amount of Tweets downloaded for each unique user in the data set is 1,462. The following graph shows the distribution of the total amount of Tweets across all users' timelines. 

<p align="center">
  <img src="/img/tweet_distribution.png" width="500"/>
</p>

Users who have less than 5 Tweets on their timeline are considered inactive and are omitted from the data set. Users who have restricted access to their timeline are regarded as not having any Tweets on their timeline. Omitting these users removes 1,286 users from the data set leaving 23,552 unique users. The resulting minimum clique size is 2, the average clique size is 5 and the maximum clique size is 36. The resulting minimum community size is 3, the average community size is 30 and the maximum community size is 318. The following graphs represent the distribution of clique and community sizes. 

<p align="center">
  <img src="/img/clique_size_distribution.png" width="350"/><img src="/img/community_size_distribution.png" width="350"/>
</p>

## Process
### LDA Model
Latent Dirichlet Allocation, or LDA, is a generative statistical modeling approach where topics are derived from a corpus of known training data, which provides a mechanism for predicting the distribution of topics of unseen documents. 

<p align="center">
  <img src="/img/lda_model.png" width="500"/>
  <br>*Yuhao Yang, “Topic Modeling with LDA”. www.youtube.com/watch?v=ZgyA1Q2ywbM
</p>

### Overview
The goal of processing the collected data for this project is to find a way to categorize the discussions that are contained in the Tweet documents and use that information to analyze the communities. The approach will be to create an LDA topic model from a corpus of known categorical information, transform each of the user's Tweets into a vectorized distribution of topics by querying that model, and use that information to find similarities between other users. 

<p align="center">
  <img src="/img/process.png" width="500"/>
</p>

### Creating the Model
#### Training Set
When selecting a corpus as a training set for this project, the approach was to choose a collection of documents that would be good at explaining or categorizing the collected data. For example, if the source of the collected data comes from a forum about hunting, a good corpus to use to train the model might be a magazine subscription about outdoor activities. Since it's not possible to know what randomly selected Twitter users will be discussing, a more broadly defined categorical source must be selected for the corpus. In this project, the entire Wikipedia articles repository was selected. 

#### Preprocess the Training Set
The Wikipedia articles were downloaded from an FTP repository as one large compressed XML file. Gensim contains a class that can automatically extract each article from the XML file, tokenize the words in each article, and add each article to a corpus. For this project, I chose to lemmatize the words in the corpus and select only words that are nouns to be added. I also used the large stopword list from http://www.ranks.nl/stopwords to filter the corpus. 

#### Dictionary
The dictionary is derived from the corpus. It represents the total vocabulary that the model will have. I chose to limit the dictionary to 100,000 words and I also chose to omit words that occurred in less than 5 articles or more than 5% of all the articles. 

#### Model
The model was trained with the number of topics set to 100 using an asymmetric prior and requiring 5 passes through the corpus. The other parameters for training the model were set to their default values.




