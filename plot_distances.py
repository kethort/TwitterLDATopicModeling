#http://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python
import random
import click
import csv
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
import shutil
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from gensim import corpora, models, matutils

def community_user_distances(community_dir):
    '''
        for each user find the distance from every other user using their document vectors
        writes files to corresponding community directories

        This method executes quickly so everytime it is run, the files that are currently in the 
        directories are removed. 

        Dictionary <k, v>(user_id, doc_vec)
    '''
    if not os.path.exists(os.path.dirname(community_dir + '/distance_info/')):
        os.makedirs(os.path.dirname(community_dir + '/distance_info/'), 0o755)
    
    cos_file = community_dir + '/distance_info/cosine'  
    hell_file = community_dir + '/distance_info/hellinger'
    euc_file = community_dir + '/distance_info/euclidean'
    jen_shan_file = community_dir + '/distance_info/jensen_shannon'

    outfiles = [cos_file, hell_file, euc_file, jen_shan_file]

    for outfile in outfiles:
       if os.path.exists(outfile):
           os.remove(outfile)
    
    # load the community document vector dictionary from file
    with open(community_dir + '/community_doc_vecs.json', 'r') as community_doc_vecs_file:
        community_doc_vecs = json.load(community_doc_vecs_file)

    print('Calculating user distances for: ' + community_dir)
    with open(cos_file, 'a') as cosfile, open(hell_file, 'a') as hellfile, open(euc_file, 'a') as eucfile, open(jen_shan_file, 'a') as jenshanfile:
        for key in sorted(community_doc_vecs):
            user = key
            # only necessary to compare each user with another user once
            vec_1 = community_doc_vecs.pop(key)

            for key_2 in sorted(community_doc_vecs):
                vec_2 = community_doc_vecs[key_2]
                cosfile.write('{}\t{}\t{}\n'.format(user, key_2, distance.cosine(vec_1, vec_2)))
                hellfile.write('{}\t{}\t{}\n'.format(user, key_2, hellinger_distance(vec_1, vec_2)))
                eucfile.write('{}\t{}\t{}\n'.format(user, key_2, distance.euclidean(vec_1, vec_2)))
                jenshanfile.write('{}\t{}\t{}\n'.format(user, key_2, jensen_shannon_divergence(vec_1, vec_2)))
    community_average_distances(community_dir)

# https://gist.github.com/larsmans/3116927
def hellinger_distance(P, Q):
    return distance.euclidean(np.sqrt(np.array(P)), np.sqrt(np.array(Q))) / np.sqrt(2)

# http://stackoverflow.com/questions/15880133/jensen-shannon-distance
def jensen_shannon_divergence(P, Q):
    _P = np.array(P) / norm(np.array(P), ord=1)
    _Q = np.array(Q) / norm(np.array(Q), ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def community_average_distances(community_dir):
    tot_rows = 0
    fieldnames = ['user_1', 'user_2', 'distance']
        
    if os.path.exists(community_dir + '/distance_info/community_average_distances'):
        os.remove(community_dir + '/distance_info/community_average_distances')
    
    # load the dictionary containing the user document vectors of the community 
    with open(community_dir + '/community_doc_vecs.json', 'r') as community_doc_vecs_file:
        community_doc_vecs = json.load(community_doc_vecs_file)
       
    # access the distance files in the community directory
    for path, dirs, files in os.walk(community_dir + '/distance_info/'):
        for distance_file in files:
            # find the average distances between users for the community
            with open(path + distance_file, 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                distances = [float(row['distance']) for row in csv_reader if row['distance']]
            if distances:
                with open(path + 'community_average_distances', 'a') as outfile:
                    outfile.write('{}\t{}\n'.format(str(distance_file), scipy.mean(distances)))
        break

def user_to_internal_users_graph(community):
    '''

    creates graph displaying each user in the community comparing the jensen 
    shannon divergences against other users in same community

    Methods used:
    > internal_graph_axes()
    > draw_scatter_graph()

    '''
    # skip communities not contain any users because of earlier pre-processing
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    fieldnames = ['user_1', 'user_2', 'distance']

    jsd_path = community + '/user_to_internal_users_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    print('Drawing user to community graphs for: ' + str(community))
    for user in comm_doc_vecs:
        if not os.path.exists(jsd_path + user + '.png'):
            with open(community + '/distance_info/jensen_shannon', 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                x_axis, y_axis = internal_graph_axes(user, csv_reader, jsd_path + user)
                draw_scatter_graph(user, 'Community users', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + 0.1), jsd_path + user)

def internal_graph_axes(user, csv_reader, output_path):
    x = 0
    x_axis = []
    y_axis = []
    for row in csv_reader:
        if(row['user_1'] == user or row['user_2'] == user):
            x = x + 1
            x_axis.append(x)
            y_axis.append(float(row['distance']))
            with open(output_path, 'a') as graph_numbers:
                if(row['user_1'] == user):
                    graph_numbers.write('{}\t{}\t{}\n'.format(row['user_1'], row['user_2'], row['distance']))
                else:
                    graph_numbers.write('{}\t{}\t{}\n'.format(row['user_2'], row['user_1'], row['distance']))
    return x_axis, y_axis

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

def user_to_external_users_graph(user_topics_dir, community):
    '''

    creates graph displaying each user in the community comparing the 
    jensen shannon divergences against randomly selected users from outside
    communities.

    The measured divergences between users is averaged over a set amount of iterations.

    Methods used:
    > get_rand_users()
    > draw_scatter_graph()
    > jensen_shannon_divergence() from tweets_on_LDA.py
    > hellinger_distance() from tweets_on_LDA.py

    '''
    NUM_ITER = 10
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    # skip cliques/communities of size 1 or less
    if(len(comm_doc_vecs) <= 1):
        return

    with open(user_topics_dir + 'all_community_doc_vecs.json', 'r') as all_community_doc_vecs_file:
        all_community_doc_vecs = json.load(all_community_doc_vecs_file)

    jsd_path = community + '/user_to_external_users_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    x_axis = np.arange(1, len(comm_doc_vecs))
    external_users = get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER)
    user_avg_dist = defaultdict(list)
    fieldnames = ['user', 'rand_user', 'dist']

    print('Drawing user to external users graphs for: ' + str(community))
    for user in comm_doc_vecs:
        if not(os.path.exists(jsd_path + user)):
            y_axis = []
            i = 0
            distances = defaultdict(lambda: [0] * (len(comm_doc_vecs) - 1))
            while(i < (len(comm_doc_vecs) - 1) * NUM_ITER):
                for n in xrange(0, len(comm_doc_vecs) - 1):
                    # circular queue exceeds limitation of possible external community users
                    rand_user = external_users.pop()
                    external_users.insert(0, rand_user)
                    i += 1

                    jsd = jensen_shannon_divergence(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
                    hel = hellinger_distance(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
                    cos = distance.cosine(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
                    euc = distance.euclidean(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])

                    distances['hel'][n] += hel
                    distances['cos'][n] += cos
                    distances['euc'][n] += euc
                    distances['jen'][n] += jsd
            i = 0

            # mmmmm....repeating code...doh!
            with open(jsd_path + user, 'w') as graph_numbers:
                for dist in distances['jen']:
                    graph_numbers.write('{}\t{}\t{}\n'.format(user, 'random user', dist/NUM_ITER))
                    y_axis.append(dist/NUM_ITER)
            user_avg_dist['jen'].append(np.average(y_axis))
            if not os.path.exists(jsd_path + user + '.png'):
                draw_scatter_graph(user, 'External Users', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + .01), jsd_path + user)
            y_axis = []
            for dist in distances['hel']:
                y_axis.append(dist/NUM_ITER)
            user_avg_dist['hel'].append(np.average(y_axis))
            y_axis = []
            for dist in distances['cos']:
                y_axis.append(dist/NUM_ITER)
            user_avg_dist['cos'].append(np.average(y_axis))
            y_axis = []
            for dist in distances['euc']:
                y_axis.append(dist/NUM_ITER)
            user_avg_dist['euc'].append(np.average(y_axis))

        else:
            distances = []
            with open(jsd_path + user, 'r') as dist_file:
                csv_reader = csv.DictReader(dist_file, delimiter='\t', fieldnames=fieldnames)
                for row in csv_reader:
                    distances.append(float(row['dist']))
            user_avg_dist['jen'].append(np.average(distances))

    with open(community + '/distance_info/external_average_distances', 'w') as outfile:
        for metric in user_avg_dist:
            if(metric == 'jen'):
                outfile.write('{}\t{}\n'.format('jensen_shannon', np.average(user_avg_dist[metric])))
            elif(metric == 'hel'):
                outfile.write('{}\t{}\n'.format('hellinger', np.average(user_avg_dist[metric])))
            elif(metric == 'euc'):
                outfile.write('{}\t{}\n'.format('euclidean', np.average(user_avg_dist[metric])))
            elif(metric == 'cos'):
                outfile.write('{}\t{}\n'.format('cosine', np.average(user_avg_dist[metric])))

def get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER):
    '''

    returns multiple of a list of random users not in the current users' community

    If number of iterations is set to 10, the random users returned is equal to:
      10 * (len(users in the community) - 1)

    When number of iterations is set to a large number, the amount of possible external
    users to compare against is limited by size of length of all users in all external
    communities.

    '''

    max_external_users = len(all_community_doc_vecs) - len(comm_doc_vecs)
    external_users = []
    while True:
        rand_external_user = random.sample(list(all_community_doc_vecs), 1)[0]
        if rand_external_user not in set(external_users) and not any(rand_external_user == user for user in comm_doc_vecs):
            external_users.append(rand_external_user)
        if(len(external_users) == (len(comm_doc_vecs) - 1) * NUM_ITER or len(external_users) == max_external_users):
            return external_users

def user_internal_external_distance(community):
    '''

    folders and figures for user internal vs external distance graphs

    Methods used:
    > draw_user_internal_external_graph()

    '''
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    # skip cliques/communities of size 1 or less
    if(len(comm_doc_vecs) <= 1):
        return

    jsd_path = community + '/distance_difference_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    print('Drawing distance within community vs distance outside community for: ' + str(community))
    for user in comm_doc_vecs:
        draw_user_internal_external_graph(user, jsd_path, community, 'jensen_shannon')

def draw_user_internal_external_graph(user, dist_path, community, metric):
    '''

    user to internal against user to external distance
    graphs, puts plotted data into community directories

    '''

    if not os.path.exists(dist_path + user + '.png'):
        y_axis = []
        fieldnames = ['user_1', 'user_2', 'distance']
        output_path = dist_path + user
        with open(community + '/user_to_internal_users_graphs/' + metric + '/' + user, 'r') as dist_file:
            csv_reader = csv.DictReader(dist_file, delimiter='\t', fieldnames=fieldnames)
            for row in csv_reader:
                with open(dist_path + user, 'a') as outfile:
                    outfile.write('{}\t{}\t{}\n'.format(row['user_1'], row['user_2'], row['distance']))
                y_axis.append(float(row['distance']))
        plt.plot(np.arange(0, len(y_axis)), y_axis, 'b')
        y_axis = []
        with open(community + '/user_to_external_users_graphs/' + metric + '/' + user, 'r') as dist_file:
            csv_reader = csv.DictReader(dist_file, delimiter='\t', fieldnames=fieldnames)
            for row in csv_reader:
                with open(dist_path + user, 'a') as outfile:
                    outfile.write('{}\t{}\t{}\n'.format(row['user_1'], row['user_2'], row['distance']))
                y_axis.append(float(row['distance']))
        plt.plot(np.arange(0, len(y_axis)), y_axis, 'g')

        if metric == 'jensen_shannon':
            plt.ylabel('Divergence')
            plt.title('Divergence from ' + user + ' to Internal/External Users')
            plt.ylim([0, np.log(2)])
        else:
            plt.ylabel('Distance')
            plt.title('Distance from ' + user + ' to Internal/External Users')
            plt.ylim([0, 1])

        plt.xlabel('Users')
        plt.xlim([0, len(y_axis)])
        plt.legend(['Internal', 'External'], loc='upper left')
        plt.savefig(output_path)
        plt.close()

def num_users_distance_range_graph(community):
    '''

    directories for user to internal plus external distance graphs of community

    Methods used:
    > draw_num_users_distance_range_graph

    '''
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    # skip cliques/communities of size 1 or less
    if(len(comm_doc_vecs) <= 1):
        return

    jsd_path = community + '/num_users_distance_range_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    draw_num_users_distance_range_graph(community, comm_doc_vecs, jsd_path, 'jensen_shannon')

def draw_num_users_distance_range_graph(community, comm_doc_vecs, output_dir, metric):
    '''

    bar graph showing number of occurrences that users in community are distant 
    from other users in the same community 

    Methods used:
    > num_internal_users()
    > num_external_users()

    '''
    internal_data = community + '/user_to_internal_users_graphs/' + metric + '/'
    external_data = community + '/user_to_external_users_graphs/' + metric + '/'

    print('Drawing number users as grouped ' + str(metric) + ' distance graph for: ' + str(community))
    for user in comm_doc_vecs:
        if not os.path.exists(output_dir + user + '.png'):
            with open(internal_data + user, 'r') as plot_file:
                num_internal_users = get_num_users(plot_file, np.arange(0, 0.7, 0.1))
            with open(external_data + user, 'r') as plot_file:
                num_external_users = get_num_users(plot_file, np.arange(0, 0.7, 0.1))

            plt.xlabel('Jensen Shannon Divergence')
            plt.title('Internal/External Divergence Range from User: ' + str(user))
            objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')

            width = 0.4
            x_axis = np.arange(len(objects))
            plt.bar(x_axis - 0.2, num_internal_users, width, alpha=0.4, color='r')
            plt.bar(x_axis + 0.2, num_external_users, width, alpha=0.4, color='b')
            plt.ylabel('Number of users')
            plt.legend(['Internal', 'External',], loc='upper left')
            plt.xticks(x_axis, objects, rotation=45, fontsize='small')
            plt.subplots_adjust(bottom=0.2)
            plt.xlim([-1, len(objects)])
            plt.ylim([0, len(comm_doc_vecs)])
            plt.savefig(output_dir + user)
            plt.close()

def get_num_users(input_file, distance_range):
    '''

    list of integers showing divergences that fall within a set of ranges
    jensen shannon 

    '''
    fieldnames = ['user_1', 'user_2', 'distance']

    csv_reader = csv.DictReader(input_file, delimiter='\t', fieldnames=fieldnames)

    y_vals = [0] * len(distance_range)
    for i in range(0, len(distance_range)):
        for row in csv_reader:
            distance = float(row['distance'])
            if(i == len(distance_range) - 1):
                if(distance >= distance_range[i]):
                    y_vals[i] += 1
            else:
                if(distance >= distance_range[i] and distance < distance_range[i + 1]):
                    y_vals[i] += 1
        input_file.seek(0)

    return y_vals 

def community_average_internal_external_distance(user_topics_dir):
    '''

    graphs displaying average internal vs average external distances
    for communities and cliques

    Methods used:
    > community_clique_average_axes()

    '''
    dist_dirs = []
    clq_dists = defaultdict(list)
    comm_dists = defaultdict(list)

    for path, dirs, files in os.walk(user_topics_dir):
        for community in sorted(dirs):
            dist_dirs.append(path + community + '/distance_info/')
        break # restrict depth of folder traversal to 1

    int_clq_y_axis, int_comm_y_axis = community_clique_average_axes(dist_dirs, 'community_average_distances')
    ext_clq_y_axis, ext_comm_y_axis = community_clique_average_axes(dist_dirs, 'external_average_distances')

    plt.plot(np.arange(0, len(int_comm_y_axis)), int_comm_y_axis, 'b')
    plt.plot(np.arange(0, len(ext_comm_y_axis)), ext_comm_y_axis, 'g', alpha=0.7)
    plt.ylabel('Average Jensen Shannon Divergence')
    plt.xlabel('Community')
    plt.title('Average Internal/External Community Divergence')
    output_path = user_topics_dir + 'community_internal_external_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_comm_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

    plt.plot(np.arange(0, len(int_clq_y_axis)), int_clq_y_axis, 'b')
    plt.plot(np.arange(0, len(ext_clq_y_axis)), ext_clq_y_axis, 'g', alpha=0.7)
    plt.ylabel('Average Jensen Shannon Divergence')
    plt.xlabel('Clique')
    plt.title('Average Internal/External Clique Divergence')
    output_path = user_topics_dir + 'clique_internal_external_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_clq_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

    plt.plot(np.arange(0, len(int_clq_y_axis)), int_clq_y_axis, 'b')
    plt.plot(np.arange(0, len(int_comm_y_axis)), int_comm_y_axis, 'g', alpha=0.7)
    plt.ylabel('Average Jensen Shannon Divergence')
    plt.xlabel('Clique/Community')
    plt.title('Average Internal Clique/Community Divergence')
    output_path = user_topics_dir + 'clique_community_internal_divergence'
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, len(int_comm_y_axis) - 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

def community_clique_average_axes(dist_dirs, filename):
    '''

    two lists one with the coordinates for the y axis of the
    average distances for a clique and the other for a community.


    '''
    fieldnames = ['metric', 'distance']

    clq_y_axis = []
    comm_y_axis = []

    for community in dist_dirs:
        with open(community + filename, 'r') as avg_dist_file:
            csv_reader = csv.DictReader(avg_dist_file, delimiter='\t', fieldnames=fieldnames)
            for row in csv_reader:
                if(row['metric'] == 'jensen_shannon'):
                    if 'clique' in community:
                        clq_y_axis.append(float(row['distance']))
                    else:
                        comm_y_axis.append(float(row['distance']))
    return clq_y_axis, comm_y_axis

def average_similarity_clique_community_size_graph(user_topics_dir):
    '''

    graph displaying the average distances compared to community size
    sets up the axes for drawing other graphs

    ** currently only draws graphs using jensen shannon divergence

    Methods used:
    > get_num_communities_jsd()
    > draw_num_communities_range_graph()
    > draw_num_clq_comm_int_range_graph()
    > distributed_average_similarity_clique_community_size_graph()

    '''
    clq_x_axis = []
    clq_y_axis = []
    comm_x_axis = []
    comm_y_axis = []
    int_clq_dists_range = [0] * 7
    ext_clq_dists_range = [0] * 7
    int_comm_dists_range = [0] * 7
    ext_comm_dists_range = [0] * 7
    clq_divs = defaultdict(list)
    comm_divs = defaultdict(list)

    fieldnames = ['metric', 'distance']
    print('Drawing clique/community size to average distance graph using Jensen Shannon divergence')

    for path, dirs, files in os.walk(user_topics_dir):
        for community in sorted(dirs):
            with open(path + community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
                comm_doc_vecs = json.load(comm_doc_vecs_file)

            with open(path + community + '/distance_info/community_average_distances', 'r') as avg_dist_file:
                csv_reader = csv.DictReader(avg_dist_file, delimiter='\t', fieldnames=fieldnames)
                for row in csv_reader:
                    if(row['metric'] == 'jensen_shannon'):
                        if 'clique' in community:
                            clq_divs[len(comm_doc_vecs)].append(float(row['distance']))
                            clq_y_axis.append(float(row['distance']))
                            clq_x_axis.append(len(comm_doc_vecs))
                            int_clq_dists_range = get_num_communities_jsd(float(row['distance']), int_clq_dists_range)
                        else:
                            comm_divs[len(comm_doc_vecs)].append(float(row['distance']))
                            comm_y_axis.append(float(row['distance']))
                            comm_x_axis.append(len(comm_doc_vecs))
                            int_comm_dists_range = get_num_communities_jsd(float(row['distance']), int_comm_dists_range)

            with open(path + community + '/distance_info/external_average_distances', 'r') as ext_dist_file:
                csv_reader = csv.DictReader(ext_dist_file, delimiter='\t', fieldnames=fieldnames)
                for row in csv_reader:
                    if(row['metric'] == 'jensen_shannon'):
                        if 'clique' in community:
                            ext_clq_dists_range = get_num_communities_jsd(float(row['distance']), ext_clq_dists_range)
                        else:
                            ext_comm_dists_range = get_num_communities_jsd(float(row['distance']), ext_comm_dists_range)

        draw_num_communities_range_graph(user_topics_dir, int_clq_dists_range, ext_clq_dists_range, int_comm_dists_range, ext_comm_dists_range)
        draw_num_clq_comm_int_range_graph(user_topics_dir, int_clq_dists_range, int_comm_dists_range, 'Clique & Community Internal Divergence Distribution', 'Clique', 'Community', 'clq_comm_int_dist_avg_range')
        draw_num_clq_comm_int_range_graph(user_topics_dir, ext_clq_dists_range, ext_comm_dists_range, 'Clique & Community External Divergence Distribution', 'Clique', 'Community', 'clq_comm_ext_dist_avg_range')
        draw_num_clq_comm_int_range_graph(user_topics_dir, int_clq_dists_range, ext_clq_dists_range, 'Clique Internal & External Divergence Distribution', 'Internal', 'External', 'clq_int_ext_dist_avg_range')
        draw_num_clq_comm_int_range_graph(user_topics_dir, int_comm_dists_range, ext_comm_dists_range, 'Community Internal & External Divergence Distribution', 'Internal', 'External', 'comm_int_ext_dist_avg_range')
        distributed_average_similarity_clique_community_size_graph(user_topics_dir, clq_divs, comm_divs)

        plt.ylabel('Average Jensen Shannon divergence')
        plt.xlabel('Size of Community')
        plt.title('Average Community Similarity')
        output_path = user_topics_dir + 'community_size_average_divergence'
        plt.scatter(comm_x_axis, comm_y_axis)
        plt.ylim([0, np.log(2) + .001])
        plt.xlim([0, max(comm_x_axis) + 1])
        plt.savefig(output_path)
        plt.close()

        plt.ylabel('Average Jensen Shannon Divergence')
        plt.xlabel('Size of Clique')
        plt.title('Average Clique Divergence')
        output_path = user_topics_dir + 'clique_size_average_divergence'
        plt.scatter(clq_x_axis, clq_y_axis)
        plt.ylim([0, np.log(2) + .001])
        plt.xlim([0, max(clq_x_axis) + 1])
        plt.savefig(output_path)
        plt.close()

        break # restrict depth of folder traversal to 1

def draw_num_communities_range_graph(user_topics_dir, int_clq, ext_clq, int_comm, ext_comm):
    '''

    graph displaying internal and external distribution for average distances/divergences
    for all communities and cliques in dataset

    ** currently only displays results for jensen shannon divergence

    '''
    width = 0.2
    fig, ax = plt.subplots()
    objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')
    x_axis = np.arange(len(objects))
    rects_1 = ax.bar(x_axis, int_clq, width, color='r', align='center')
    rects_2 = ax.bar(x_axis + width, ext_clq, width, color='g', align='center')
    rects_3 = ax.bar(x_axis + width * 2, int_comm, width, color='b', align='center')
    rects_4 = ax.bar(x_axis + width * 4, ext_comm, width, color='y', align='center')
    ax.set_xlabel('Jensen Shannon Divergence Distribution')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Clique/Community Divergence Distribution')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(objects, ha='center')
    box = ax.get_position()
    ax.legend((rects_1[0], rects_2[0], rects_3[0], rects_4[0]), ('Clq Int', 'Cliq Ext', 'Comm Int', 'Comm Ext'), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(user_topics_dir + 'clq_comm_distributed_avg_range')
    plt.close(fig)

def get_num_communities_jsd(distance, avg_dists_range):
    '''

    list of integers showing the amount of cliques/communities whose
    average distances/divergences fall within a set of ranges

    '''
    grp_1 = grp_2 = grp_3 = grp_4 = grp_5 = grp_6 = grp_7 = 0
    if(distance >= 0 and distance < 0.1):
        grp_1 += 1
    elif(distance >= 0.1 and distance < 0.2):
        grp_2 += 1
    elif(distance >= 0.2 and distance < 0.3):
        grp_3 += 1
    elif(distance >= 0.3 and distance < 0.4):
        grp_4 += 1
    elif(distance >= 0.4 and distance < 0.5):
        grp_5 += 1
    elif(distance >= 0.5 and distance < 0.6):
        grp_6 += 1
    elif(distance >= 0.6):
        grp_7 += 1

    num_users = (grp_1, grp_2, grp_3, grp_4, grp_5, grp_6, grp_7)
    return np.sum([num_users, avg_dists_range], axis=0)

def draw_num_clq_comm_int_range_graph(user_topics_dir, dists_1, dists_2, title, lg_lbl_1, lg_lbl_2, out_name):
    '''

    graph displaying difference in distribution of internal average clique vs average community
    distances/divergences which fall into set of ranges

    ** currently only displays distribution for jensen shannon divergence 

    '''
    width = 0.3
    fig, ax = plt.subplots()
    objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')
    x_axis = np.arange(len(objects))
    rects_1 = ax.bar(x_axis, dists_1, width, color='r', align='center')
    rects_2 = ax.bar(x_axis + width, dists_2, width, color='b', align='center')
    ax.set_xlabel('Jensen Shannon Divergence Distribution')
    ax.set_ylabel('Number of Communities')
    ax.set_title(title)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(objects, ha='center')
    box = ax.get_position()
    ax.legend((rects_1[0], rects_2[0]), (lg_lbl_1, lg_lbl_2), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(user_topics_dir + out_name)
    plt.close(fig)

def distributed_average_similarity_clique_community_size_graph(user_topics_dir, clq_divs, comm_divs):
    '''

    graph displays the average of the internal clique and internal
    community average distance/divergence compared to the size of the clique or community

    ** currently only displays the jensen shannon divergence

    ''' 
    clq_x_axis = []
    clq_y_axis = []
    comm_x_axis = []
    comm_y_axis = []
    print('Drawing clique/community size to distributed average divergence graph')

    plt.ylabel('Average Jensen Shannon Divergence\n')
    plt.xlabel('Size of Community/Clique')
    plt.title('Average Community/Clique Similarity Distribution')

    for clq_size in clq_divs:
        clq_x_axis.append(clq_size)
        clq_y_axis.append(np.mean(clq_divs[clq_size]))

    for comm_size in comm_divs:
        comm_x_axis.append(comm_size)
        comm_y_axis.append(np.mean(comm_divs[comm_size]))

    output_path = user_topics_dir + 'clique_community_size_distributed_average_divergence'
    plt.scatter(clq_x_axis, clq_y_axis, marker='x', color='g')
    plt.scatter(comm_x_axis, comm_y_axis, marker='.', color='b')
    plt.legend(['clique', 'community',], loc='upper center', ncol=2)
    plt.ylim([0, np.log(2) + .001])
    plt.xlim([0, max(comm_x_axis) + 1])
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path)
    plt.close()

def draw_topic_distribution_graph(comm_doc_vecs, user, output_path):
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

def user_topic_distribution(community_dir):
    print('Getting topic distribution for : ' + community_dir)
    output_path = community_dir + '/topic_distribution_graphs/'
            
    if not os.path.exists(os.path.dirname(output_path)):
    	os.makedirs(os.path.dirname(output_path), 0o755)

    with open(community_dir + '/community_doc_vecs.json', 'r') as infile:
        comm_doc_vecs = json.load(infile)

    for user in comm_doc_vecs:
        draw_topic_distribution_graph(comm_doc_vecs, user, output_path)

def delete_inactive_communities(community):
    '''

    deletes clique and corresponding community directories with 1 or less 
    active users

    '''
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    if len(comm_doc_vecs) <= 1:
        if os.path.exists(community):
            print('removing: ' + str(community))
            shutil.rmtree(community)
        if os.path.exists(community.replace('clique', 'community')):
            shutil.rmtree(community.replace('clique', 'community'))

def delete_inactive_users(community):
    '''

    removes users from serialized community document vector dictionaries
    if the amount of times they tweeted is less than 10

    '''
    try:
        with open(community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
            comm_doc_vecs = json.load(comm_doc_vecs_file)
    except:
        comm_doc_vecs = {}

    for user in comm_doc_vecs.copy():
        count = 0
        with open('dnld_tweets/' + user, 'r') as tweet_file:
            count = sum(1 for line in tweet_file if line.strip())
        if(count < 10):
            del comm_doc_vecs[user]
          
    with open(community + '/community_doc_vecs.json', 'w') as comm_doc_vecs_file:
        json.dump(comm_doc_vecs, comm_doc_vecs_file, sort_keys=True, indent=4)
    
def dir_to_iter(user_topics_dir):
     for path, dirs, files in os.walk(user_topics_dir):
         for community in sorted(dirs):
             yield(path + community)
         break

def main(user_topics_dir):
    '''
    argument for program should be user_topics_dir location
    
    example:
        - python plot_distances.py user_topics_75/

    '''
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
	
    pool.map(delete_inactive_users, dir_to_iter(user_topics_dir))
    pool.map(delete_inactive_communities, dir_to_iter(user_topics_dir))

    pool.map(community_user_distances, dir_to_iter(user_topics_dir))

    pool.map(user_to_internal_users_graph, dir_to_iter(user_topics_dir)) 

    func = partial(user_to_external_users_graph, user_topics_dir)
    pool.map(func, dir_to_iter(user_topics_dir))
 
    pool.map(user_internal_external_distance, dir_to_iter(user_topics_dir))

    pool.map(num_users_distance_range_graph, dir_to_iter(user_topics_dir))

    pool.map(user_topic_distribution, dir_to_iter(user_topics_dir))

    pool.terminate()
 
    community_average_internal_external_distance(user_topics_dir)
    average_similarity_clique_community_size_graph(user_topics_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
