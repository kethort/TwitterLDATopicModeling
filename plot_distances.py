import random
import csv
import pandas as pd
import json
import os
import ast
import sys
import scipy
from scipy.spatial import distance
from scipy.linalg import norm
from scipy.stats import entropy
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from gensim import corpora, models, matutils

def distance_from_user_to_internal_users(community):
    '''
        for each user find the distance from every other user using their probability distribution vectors

        This method executes quickly so everytime it is run the older files are overwritten 
        Dictionary <k, v>(user_id, distribution_vector)

    '''
    if not os.path.exists(os.path.dirname(community + '/distance_info/')):
        os.makedirs(os.path.dirname(community + '/distance_info/'), 0o755)
    
    cos_file = community + '/distance_info/cosine'  
    hell_file = community + '/distance_info/hellinger'
    euc_file = community + '/distance_info/euclidean'
    jen_shan_file = community + '/distance_info/jensen_shannon'

    outfiles = [cos_file, hell_file, euc_file, jen_shan_file]

    for outfile in outfiles:
       if os.path.exists(outfile):
           os.remove(outfile)
    
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    print('Calculating user distances for: ' + community)
    with open(cos_file, 'a') as cosfile, open(hell_file, 'a') as hellfile, open(euc_file, 'a') as eucfile, open(jen_shan_file, 'a') as jenshanfile:
        for key in sorted(comm_doc_vecs):
            user = key
            # only necessary to compare each user with any other user once
            vec_1 = comm_doc_vecs.pop(key)

            for key_2 in sorted(comm_doc_vecs):
                vec_2 = comm_doc_vecs[key_2]
                cosfile.write('{}\t{}\t{}\n'.format(user, key_2, distance.cosine(vec_1, vec_2)))
                hellfile.write('{}\t{}\t{}\n'.format(user, key_2, hellinger_distance(vec_1, vec_2)))
                eucfile.write('{}\t{}\t{}\n'.format(user, key_2, distance.euclidean(vec_1, vec_2)))
                jenshanfile.write('{}\t{}\t{}\n'.format(user, key_2, jensen_shannon_divergence(vec_1, vec_2)))
    median_community_distances(community)

def hellinger_distance(P, Q):
    return distance.euclidean(np.sqrt(np.array(P)), np.sqrt(np.array(Q))) / np.sqrt(2)

def jensen_shannon_divergence(P, Q):
    _P = np.array(P) / norm(np.array(P), ord=1)
    _Q = np.array(Q) / norm(np.array(Q), ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def median_community_distances(community):
    '''
    calculates and stores the median JSD for each community into a file
    
    '''
    fieldnames = ['user_1', 'user_2', 'distance']
    if os.path.exists(community + '/distance_info/median_community_distances'):
        os.remove(community + '/distance_info/median_community_distances')
    
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
       
    print('Calculating median distances for: ' + community)
    for path, dirs, files in os.walk(community + '/distance_info/'):
        for distance_file in files:
            with open(path + distance_file, 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                distances = [float(row['distance']) for row in csv_reader if row['distance']]
            if distances:
                with open(path + 'median_community_distances', 'a') as outfile:
                    outfile.write('{}\t{}\n'.format(str(distance_file), np.median(distances)))
        break

def user_to_internal_users_graph(community):
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences between
    their probability distribution vectors against other users in same community

    x-axis: users in community, y-axis: distance from observed user 
    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    if(len(comm_doc_vecs) <= 1):
        return

    jsd_path = community + '/user_to_internal_users_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    print('Drawing internal distance graphs for community: ' + str(community))
    x_axis = np.arange(1, len(comm_doc_vecs))
    df = pd.read_csv(community + '/distance_info/jensen_shannon', sep='\t', header=None, names=['user_1', 'user_2', 'distance'])
    for user in comm_doc_vecs:
        if not os.path.exists(jsd_path + user + '.png'):
            new_df = df[(df.user_1 == int(user)) | (df.user_2 == int(user))]
            new_df.to_csv(jsd_path + str(user), sep='\t', header=None, index=None)
            y_axis = new_df['distance'].tolist()
            draw_scatter_graph(user, 'Community Members', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + 0.1), jsd_path + user)

def draw_scatter_graph(title, x_label, y_label, x_axis, y_axis, min_x, max_x, min_y, max_y, output_path):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x_axis, y_axis, 'o')
    ax.axis([min_x, max_x, min_y, max_y])
    plt.margins(0.2)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close(fig)

def user_to_external_users_graph(working_dir, community):
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences 
    against randomly selected users from outside communities.

    x-axis: users outside of community, y-axis: distance from observed user 

    '''
    NUM_ITER = 10
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    if(len(comm_doc_vecs) <= 1):
        return

    with open(working_dir + 'all_community_doc_vecs.json', 'r') as all_community_doc_vecs_file:
        all_community_doc_vecs = json.load(all_community_doc_vecs_file)

    jsd_path = community + '/user_to_external_users_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    x_axis = np.arange(1, len(comm_doc_vecs))
    external_users = get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER)
    external_median_jsd = []

    print('Drawing user to external users graphs for: ' + str(community))
    for user in comm_doc_vecs:
        if not(os.path.exists(jsd_path + user)):
            y_axis = []
            i = 0
            jsd = [0] * (len(comm_doc_vecs) - 1)
            # running time is uffed like a beach here
            while(i < (len(comm_doc_vecs) - 1) * NUM_ITER):
                for n in range(0, len(comm_doc_vecs) - 1):
                    # circular queue because it's possible to exceed amount of all users in entire dataset
                    rand_user = external_users.pop()
                    external_users.insert(0, rand_user)
                    jsd[n] += jensen_shannon_divergence(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
                    i += 1

            with open(jsd_path + user, 'w') as graph_numbers:
                for div in jsd:
                    graph_numbers.write('{}\t{}\t{}\n'.format(user, 'random user', div/NUM_ITER))
                    y_axis.append(div/NUM_ITER)
            external_median_jsd.append(np.median(y_axis))
            if not os.path.exists(jsd_path + user + '.png'):
                draw_scatter_graph(user, 'External Users', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + .01), jsd_path + user)

        else:
            df = pd.read_csv(jsd_path + user, sep='\t', header=None, names=['user', 'rand_user', 'distance'])
            external_median_jsd.append(np.median(df['distance'].tolist()))

    with open(community + '/distance_info/median_external_community_distances', 'w') as outfile:
        outfile.write('{}\t{}\n'.format('jensen_shannon', np.median(external_median_jsd)))

def get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER):
    '''
    returns multiple of a list of random users not in the current users' community

    if number of iterations is set to 10, the random users returned is equal to:
      10 * (len(users in the community) - 1)

    '''
    max_external_users = len(all_community_doc_vecs) - len(comm_doc_vecs)
    internal_users = set(user for user in comm_doc_vecs)
    external_users = []
    while True:
        rand_external_user = random.sample(list(all_community_doc_vecs), 1)[0]
        if rand_external_user not in set(external_users) and rand_external_user not in internal_users:
            external_users.append(rand_external_user)
        if(len(external_users) == (len(comm_doc_vecs) - 1) * NUM_ITER or len(external_users) == max_external_users):
            return external_users

def user_internal_external_graphs(community):
    '''
    user to internal against user to external distance
    graphs, puts plotted data into community directories

    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    if(len(comm_doc_vecs) <= 1):
        return

    jsd_path = community + '/distance_difference_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    print('Drawing internal vs external distance for: ' + str(community))
    for user in comm_doc_vecs:
        if not os.path.exists(jsd_path + user + '.png'):
            int_df = pd.read_csv(community + '/user_to_internal_users_graphs/jensen_shannon/' + user, sep='\t', header=None, names=['user_1', 'user_2', 'distance'])
            y_axis = int_df['distance'].tolist()
            plt.plot(np.arange(0, len(y_axis)), y_axis, 'b')

            ext_df = pd.read_csv(community + '/user_to_external_users_graphs/jensen_shannon/' + user, sep='\t', header=None, names=['user_1', 'user_2', 'distance'])
            y_axis = ext_df['distance'].tolist()
            plt.plot(np.arange(0, len(y_axis)), y_axis, 'b')

            pd.concat([int_df, ext_df]).to_csv(jsd_path + str(user), sep='\t', header=None, index=None)

            plt.ylabel('Divergence')
            plt.title('Divergence from ' + user + ' to Internal & External Users')
            plt.ylim([0, np.log(2)])
            plt.xlabel('Users')
            plt.xlim([0, len(y_axis) - 1])
            plt.xticks(np.arange(0, len(y_axis), 2))
            plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(jsd_path + user)
            plt.close()

def community_median_internal_external_distance_graph(working_dir):
    '''
    graphs displaying median internal vs median external distances for all the communities & cliques

    '''
    int_clq_y_axis, int_comm_y_axis = community_median_internal_external_distance_graph_y_axes(working_dir, 'median_community_distances')
    ext_clq_y_axis, ext_comm_y_axis = community_median_internal_external_distance_graph_y_axes(working_dir, 'median_external_community_distances')

    plt.plot(np.arange(0, len(int_comm_y_axis)), int_comm_y_axis, 'b')
    plt.plot(np.arange(0, len(ext_comm_y_axis)), ext_comm_y_axis, 'g', alpha=0.7)
    plt.xticks(np.arange(0, len(int_comm_y_axis), 2))
    plt.ylabel('Median Jensen Shannon Divergence')
    plt.xlabel('Community')
    plt.title('Median Internal & External Community Divergence')
    output_path = working_dir + 'community_internal_external_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_comm_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

    plt.plot(np.arange(0, len(int_clq_y_axis)), int_clq_y_axis, 'b')
    plt.plot(np.arange(0, len(ext_clq_y_axis)), ext_clq_y_axis, 'g', alpha=0.7)
    plt.xticks(np.arange(0, len(int_clq_y_axis), 2))
    plt.ylabel('Median Jensen Shannon Divergence')
    plt.xlabel('Clique')
    plt.title('Median Internal & External Clique Divergence')
    output_path = working_dir + 'clique_internal_external_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_clq_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

    plt.plot(np.arange(0, len(int_clq_y_axis)), int_clq_y_axis, 'b')
    plt.plot(np.arange(0, len(int_comm_y_axis)), int_comm_y_axis, 'g', alpha=0.7)
    plt.xticks(np.arange(0, len(int_clq_y_axis), 2))
    plt.ylabel('Median Jensen Shannon Divergence')
    plt.xlabel('Clique/Community')
    plt.title('Median Internal Clique & Community Divergence')
    output_path = working_dir + 'clique_community_internal_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_comm_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

def community_median_internal_external_distance_graph_y_axes(working_dir, filename):
    clq_y_axis = []
    comm_y_axis = []
    for path, dirs, files in os.walk(working_dir):
        for community in sorted(dirs):
            distance_dir = path + community + '/distance_info/'
            df = pd.read_csv(distance_dir + filename, sep='\t', header=None, names=['metric', 'distance'])
            if 'clique' in community:
                clq_y_axis.append(float(df[df.metric == 'jensen_shannon']['distance']))
            else:
                comm_y_axis.append(float(df[df.metric == 'jensen_shannon']['distance']))
        break # restrict depth of folder traversal to 1
    return clq_y_axis, comm_y_axis

def median_overall_internal_distance_by_community_size_graph(working_dir):
    '''
    displays the overall median internal clique and internal community divergence in relation
    to the sizes of the cliques or communities

    ''' 
    clq_dists = defaultdict(list)
    comm_dists = defaultdict(list)

    for path, dirs, files in os.walk(working_dir):
        for community in sorted(dirs):
            comm_doc_vecs = open_community_document_vectors_file(path + community + '/community_doc_vecs.json')
            df = pd.read_csv(path + community + '/distance_info/median_community_distances', sep='\t', header=None, names=['metric', 'distance'])
            if 'clique' in community:
                clq_dists[len(comm_doc_vecs)].append(float(df[df.metric == 'jensen_shannon']['distance']))
            else:
                comm_dists[len(comm_doc_vecs)].append(float(df[df.metric == 'jensen_shannon']['distance']))
        break

    clq_x_axis = []
    clq_y_axis = []
    comm_x_axis = []
    comm_y_axis = []
    print('Drawing overall median clique & community divergence by size graph')

    plt.ylabel('Median Jensen Shannon Divergence\n')
    plt.xlabel('Size of Community/Clique')
    plt.title('Median Community/Clique Similarity Distribution')

    for clq_size in clq_dists:
        clq_x_axis.append(clq_size)
        clq_y_axis.append(np.mean(clq_dists[clq_size]))

    for comm_size in comm_dists:
        comm_x_axis.append(comm_size)
        comm_y_axis.append(np.mean(comm_dists[comm_size]))

    output_path = working_dir + 'median_overall_internal_distance_by_community_size'
    plt.scatter(clq_x_axis, clq_y_axis, marker='x', color='g')
    plt.scatter(comm_x_axis, comm_y_axis, marker='.', color='b')
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, max(comm_x_axis) + 1])
    plt.legend(['Clique', 'Community'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

def user_topic_distribution_graph(community):
    '''
    Each of the user's topic probability distribution vectors can be visualized using 
    this function

    '''
    print('Getting topic distribution for : ' + community)
    output_path = community + '/topic_distribution_graphs/'
            
    if not os.path.exists(os.path.dirname(output_path)):
    	os.makedirs(os.path.dirname(output_path), 0o755)

    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    for user in comm_doc_vecs:
        if not os.path.exists(output_path + user + '.png'):
            y_axis = []
            x_axis = []
            for topic_id, dist in enumerate(comm_doc_vecs[user]):
                x_axis.append(topic_id + 1)
                y_axis.append(dist)
            width = 1 
            plt.bar(x_axis, y_axis, width, align='center', color='r')
            plt.xlabel('Topics')
            plt.ylabel('Probability')
            plt.title('Topic Distribution for User: ' + user)
            plt.xticks(np.arange(2, len(x_axis), 2), rotation='vertical', fontsize=7)
            plt.subplots_adjust(bottom=0.2)
            plt.ylim([0, np.max(y_axis) + .01])
            plt.xlim([0, len(x_axis) + 1])
            plt.savefig(output_path + user)
            plt.close()

def delete_inactive_communities(community):
    '''
    deletes clique and corresponding community directories with 1 or less active users

    if not used, most graphing functions above will omit communities of size 1 or less
    because no comparisons can be made

    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    if len(comm_doc_vecs) <= 1:
        if os.path.exists(community):
            print('removing: ' + str(community))
            shutil.rmtree(community)
        if os.path.exists(community.replace('clique', 'community')):
            shutil.rmtree(community.replace('clique', 'community'))

def delete_inactive_users(community):
    '''
    removes users from the dataset if the amount of times they tweeted is less than 10

    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')

    for user in comm_doc_vecs.copy():
        count = 0
        with open('dnld_tweets/' + user, 'r') as tweet_file:
            count = sum(1 for line in tweet_file if line.strip())
        if(count < 10):
            del comm_doc_vecs[user]
          
    with open(community + '/community_doc_vecs.json', 'w') as comm_doc_vecs_file:
        json.dump(comm_doc_vecs, comm_doc_vecs_file, sort_keys=True, indent=4)

def open_community_document_vectors_file(file_name):
    try:
        with open(file_name, 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}
    return comm_doc_vecs

    
def dir_to_iter(working_dir):
    for path, dirs, files in os.walk(working_dir):
        for community in dirs:
            yield(path + community)
        break

def main(working_dir):
    '''
    argument for program should be working_dir location
    
    example:
        - python plot_distances.py user_topics_75/

    '''
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
	
#    func = partial(delete_inactive_users, document_vectors)
#    pool.map(delete_inactive_users, dir_to_iter(working_dir))
#
#    pool.map(delete_inactive_communities, dir_to_iter(working_dir))

    pool.map(distance_from_user_to_internal_users, dir_to_iter(working_dir))

    pool.map(user_to_internal_users_graph, dir_to_iter(working_dir)) 

    func = partial(user_to_external_users_graph, working_dir)
    pool.map(func, dir_to_iter(working_dir))

    pool.map(user_internal_external_graphs, dir_to_iter(working_dir))

    pool.map(user_topic_distribution_graph, dir_to_iter(working_dir))

    pool.terminate()
 
    community_median_internal_external_distance_graph(working_dir)
    median_overall_internal_distance_by_community_size_graph(working_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
