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

## Preliminary
### LDA Model
Latent Dirichlet Allocation, or LDA, is a generative statistical modeling approach where topics are derived from a corpus of known training data, which provides a mechanism for predicting the distribution of topics of unseen documents. 

<p align="center">
  <img src="/img/lda_model.png" width="500"/>
  <br>*Yuhao Yang, “Topic Modeling with LDA”. www.youtube.com/watch?v=ZgyA1Q2ywbM
</p>

### Overview
The goal of processing the collected data for this project is to find a way to categorize the discussions that are contained in the Tweet documents and use that information to analyze the communities. The approach will be to create an LDA topic model from a corpus of known categorical information, transform each of the user's Tweets into a vectorized distribution of topics by querying that model, and use that information to find similarities between other users. 

<p align="center">
  <img src="/img/lda_architecture.png" width="500" height="600"/>
</p>

### Creating the Model
#### Training Set
When selecting a corpus as a training set for this project, the approach was to choose a collection of documents that would be good at explaining or categorizing the collected data. For example, if the source of the collected data comes from a forum about hunting, a good corpus to use to train the model might be a magazine subscription about outdoor activities. Since it's not possible to know what randomly selected Twitter users will be discussing, a more broadly defined categorical source must be selected for the corpus. In this project, the entire Wikipedia articles repository was selected. 

#### Preprocess the Training Set
The Wikipedia articles were downloaded from an FTP repository as one large compressed XML file. Gensim contains a class that can automatically extract each article from the XML file, tokenize the words in each article, and add each article to a corpus. For this project, I chose to lemmatize the words in the corpus and select only words that are nouns to be added. I also used the large stopword list from http://www.ranks.nl/stopwords to filter the corpus. 

#### Preprocess Documents of Interest
All of a single users tweets make up one document. The words in each document are filtered using a stopword list. URL‘s and special characters are removed, and the words are lemmatized.

#### Dictionary
The dictionary is derived from the corpus. It represents the total vocabulary that the model will have. I chose to limit the dictionary to 100,000 words and I also chose to omit words that occurred in less than 5 articles or more than 5% of all the articles. 

#### Model
The model was trained with the number of topics set to 50, using an asymmetric prior and requiring 5 passes through the corpus. The other parameters for training the model were set to their default values.

#### Inference
Each preprocessed document of interest is used to query the trained model which generates a topic probability distribution vector for the document.

#### Calculations
Using the community topology as a map, each user‘s probability distribution vector is used to calculate similarities between other users. The similarity metric used is the Jensen Shannon divergence. 

#### Visualizations
Each users‘ topic probability distribution vector is plotted and a pyLDAvis output showing the topics in the model with their top-n words is generated.

## Results
### Viewing User Topics
A randomly selected user from the dataset is examined and determined to be a bankruptcy law firm in Arizona. The topic probability distribution vector for the user is generated using the LDA model. The following figure displays the topic probability distribution for the user. The topic probability distribution vector for this Twitter user shows topics 13 and 21 to be the most prominent. 

<p align="center">
  <img src="/img/user_x_distribution.png" width="500"/>
</p>

The LDA model is visualized in the next figure using pyLDAvis, where each circle on the map is a topic in the model. The top 30 most relevant words in each topic are listed next to the map. The red bar indicates the frequency of occurrence of the word in the topic and the blue bar represents the frequency of occurrence of the word in the entire corpus. The distance between the topics on the map is defined by the Jensen Shannon Divergence and represents how closely related the topics are to each other in the model. 

<p align="center">
  <img src="/img/user_x_lda_vis.png" width="750" height="550"/>
</p>

From this LDA model visualization, topic 13 displays many words about the legal system. Although not shown, topic 21 in the model contains words about economic institutions and corporations. Both these topics describe the selected user quite well. The two topics in the LDA model visualization are also very close together on the map, indicating that the words in those topics are semantically related to each other in this model.

### Finding Distances Between Users Based on their Interests
A community of users can be evaluated by calculating the distances between their probability distribution vectors generated from the LDA model. The following figure displays the distance between one user compared to each of the other users in their community as well as the distance from randomly selected users not in the same community. 

<p align="center">
  <img src="/img/user_internal_external.png" width="500"/>
</p>

The following figure displays the separation in the median internal and external distances of all the communities in the dataset. The distance metric used in the comparisons is the Jensen Shannon Divergence which has a range from 0 to the natural log of 2. 

<p align="center">
  <img src="/img/community_median_internal_external.png" width="500"/>
</p>

## Conclusion
Using an LDA model to generate topic probability distributions for conversations on social media platforms like Twitter, gives context to the nodes in a network topology and provides a basis for verifying the strength of community detection algorithms. Although the model we trained can approximately categorize the topic of a users‘ tweets, the model had trouble identifying themes for users who either did not tweet enough or did not tweet about anything specific. The training process of an LDA model includes a myriad of different factors which can affect the outcome of the entire experiment. Due to the anatomy of Twitter conversations, the use of lemmatization and removal of hashtags possibly limited our model‘s ability to identify topics for certain users. Selecting Wikipedia articles as a corpus for our model and using a small amount of topics to train the model had the effect of creating very broad topics. Future work in this area might include creating various different models and analyzing their coherence to determine which of them could work best. In a very large dataset, there will inevitably exist many outliers and, in our experiment, we found there to be many single users and even entire communities that did not fit into our definition of how a community or user of a community should be interacting. Fortunately, a good majority of the communities we tested displayed a strong indication that the users in the communities shared similar interests. Whether or not these results indicate a strong correlation that an LDA model can be used to understand the strength of community detection algorithms is a subject for future research. 
