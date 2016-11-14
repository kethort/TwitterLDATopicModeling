import gensim
from gensim import utils, corpora, models
import ast
import re
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import tweets_on_LDA_2 as tlda
import plot_distances as pltd

# load wiki dictionary
dictionary = corpora.Dictionary.load('data/tweets.dict')

# load trained wiki model from file
lda = models.LdaModel.load('data/tweets_100_lem_5_pass.model')

def aggregate_tweets(i, clique):
    print('Aggregating tweets for community_' + str(i))
    with open('aggregated_tweets/clique_' + str(i), 'w') as outfile:
        for user in ast.literal_eval(clique):
            if os.path.exists('dnld_tweets/' + str(user)):
                with open('dnld_tweets/' + str(user)) as tweet:
                    for line in tweet:
                        outfile.write(line)

def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])
    return [(topic_id, topic_value) for topic_id, topic_value in enumerate(topic_dist)]

def doc_to_vec(tweet):
    doc = tlda.convert_to_doc(tweet)

    # create bag of words from input document
    doc_bow = dictionary.doc2bow(doc)

    # queries the document against the LDA model and associates the data with probabalistic topics
    doc_lda = get_doc_topics(lda, doc_bow)
    return gensim.matutils.sparse2full(doc_lda, lda.num_topics)

def draw_dist_graph(filename, dense_vec):
    print('Drawing probability distribution graph for ' + filename)
    if not os.path.exists(filename + '.png'):
        y_axis = []
        x_axis = []
                        
        for topic_id, dist in enumerate(dense_vec):
            x_axis.append(topic_id + 1)
            y_axis.append(dist)
        width = 1 

        plt.bar(x_axis, y_axis, width, align='center', color='r')
        plt.xlabel('Topics')
        plt.ylabel('Probability')
        plt.title('Topic Distribution for clique')
        plt.xticks(np.arange(2, len(x_axis), 2), rotation='vertical', fontsize=7)
        plt.subplots_adjust(bottom=0.2)
        plt.ylim([0, np.max(y_axis) + .01])
        plt.xlim([0, len(x_axis) + 1])
        plt.savefig(filename)
        plt.close()

def write_topn_words():
    print('Writing topn words for LDA model')
    reg_ex = re.compile('(?<![\s/])/[^\s/]+(?![\S/])')
    with open('topn_words.txt', 'w') as outfile:
        for i in range(lda.num_topics):
            outfile.write('{}\n'.format('Topic #' + str(i + 1) + ': '))
            for word, prob in lda.show_topic(i, topn=20):
                word = reg_ex.sub('', word)
                outfile.write('\t{}\n'.format(word.encode('utf-8')))
            outfile.write('\n')

def write_jsd_nums(i, clique_vec, community):
    with open('distribution_graphs/jensen_shannon_community_' + str(i), 'w') as outfile:
        for user in ast.literal_eval(community):
            if os.path.exists('dnld_tweets/' + str(user)):
                print('Writing Jensen Shannon distance for user ' + str(user) + ' in community ' + str(i))
                jsd = pltd.jensen_shannon_divergence(clique_vec, doc_to_vec('dnld_tweets/' + str(user)))
                outfile.write('{}\t{}\t{}\n'.format(user, 'clique', jsd))

def main(arg1, arg2):
    write_topn_words()

    if not os.path.exists(os.path.dirname('aggregated_tweets/')):
        os.makedirs(os.path.dirname('aggregated_tweets/'), 0o755)

    if not os.path.exists(os.path.dirname('distribution_graphs/')):
        os.makedirs(os.path.dirname('distribution_graphs/'), 0o755)

    with open(arg1, 'r') as infile:
        for i, clique in enumerate(infile):
            aggregate_tweets(i, clique)

    clique_vecs = {}
    for path, dirs, files in os.walk('aggregated_tweets/'):
        for filename in files:   
            print('Getting document vector for ' + filename)
            clique_vecs[filename] = doc_to_vec(path + filename)
            draw_dist_graph('distribution_graphs/' + filename, clique_vecs[filename])
        break

    with open('document_vectors.pickle', 'wb') as outfile:
        pickle.dump(clique_vecs, outfile)

    with open(arg2, 'r') as infile:
        for i, community in enumerate(infile):
            write_jsd_nums(i, clique_vecs['clique_' + str(i)], community)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))


