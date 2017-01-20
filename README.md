# twitter_LDA_topic_modeling
Uses the Gensim library in Python to create a corpus from documents and train an LDA model.
The model is used to evaluate similarities in conversations between Twitter users.
The groups of Twitter users being evaluated were found using community detection algorithms. Tweepy was used to communicate with the Twitter API. Matplotlib and pyLDAvis were used to produce illustrations of the data. Scipy, Numpy and NLTK were used during processing of the data. 

## Premise
The follower relationship on Twitter differs from other social media networks like Facebook because users who create friendships on Twitter don't always do so based on prior human interaction. Many Twitter users follow others based purely on shared interests. Using the Twitter API, it's possible to isolate sections of localized populations who have formed communities based on a 1:1 follower relationship between other users. These communities of users often display a central purpose or associative theme of shared interests. The goal of this project is to reinforce the integrity of community detection algorithms by attempting to associate a communitys' shared interests into latent topics, which are categorized using a trained statistical model.


<p align="center">
  <img src="img/22393440.png" width="350"/>
</p>

