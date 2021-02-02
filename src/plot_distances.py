import random
import pandas as pd
import json
import os
import ast
import sys
import csv
import argparse
import argcomplete
import scipy
import tqdm
import shutil
import ntpath
from shutil import copyfile
from scipy.linalg import norm
from scipy.stats import entropy
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models, matutils

'''
    command line tool to calculate the internal/external JSD from/to users within/outside of
    a clique/community of which the topic probability distributions should be calculated already from
    previous scripts. Uses matplotlib to plot the calculated data.

    This script was built using very specific instructions for research.
    Most of the code is commented out as it won't be for typical use.
'''

def jensen_shannon_divergence(P, Q):
    _P = np.array(P) / norm(np.array(P), ord=1)
    _Q = np.array(Q) / norm(np.array(Q), ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def calculate_internal_distances(community):
    # calculates jensen shannon divergance from every other user in the same community using their probability distribution vectors
    # calculates the median of all community member results and saves the output to file
    # saves the individual recorded divergences in file in corresponding community folder
    distance_dir = os.path.join(community, 'calculated_distances/')
    if(os.path.exists(os.path.join(distance_dir, 'median_community_distances'))): return
    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    
    if(len(comm_doc_vecs) <= 1): return
    
    if not os.path.exists(os.path.dirname(distance_dir)):
        os.makedirs(os.path.dirname(distance_dir), 0o755)

    jen_shan_file = os.path.join(distance_dir, 'jensen_shannon')
    if os.path.exists(jen_shan_file): os.remove(jen_shan_file)
    with open(jen_shan_file, 'w') as out:
        for key in sorted(comm_doc_vecs):
            user = key
            vec_1 = comm_doc_vecs.pop(key)

            for key_2 in sorted(comm_doc_vecs):
                vec_2 = comm_doc_vecs[key_2]
                out.write('{}\t{}\t{}\n'.format(user, key_2, jensen_shannon_divergence(vec_1, vec_2)))

def individual_user_distance_graphs(internal, community):
    # creates graph displaying each user in the community from the result of the calculate_internal_distances function

    # x axis: users in community, y axis: distance from observed user
    distance_dir = os.path.join(community, 'calculated_distances/')
    if internal:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon')
        out_path = os.path.join(os.path.join(community, 'internal_user_graphs/jensen_shannon/'))
        out_file = os.path.join(distance_dir, 'community_distances')
    else:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon_ext')
        out_path = os.path.join(os.path.join(community, 'external_user_graphs/jensen_shannon/'))
        out_file = os.path.join(distance_dir, 'ext_community_distances')
    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    
    if(len(comm_doc_vecs) <= 1): return
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), 0o755)
    x_axis = np.arange(1, len(comm_doc_vecs))
    df = pd.read_csv(jsd_dists, sep='\t', header=None, names=['user_1', 'user_2', 'distance'])
    for user in comm_doc_vecs:
        if not os.path.exists(os.path.join(out_path, user + '.png')):
            new_df = df[(df.user_1 == int(user)) | (df.user_2 == int(user))]
            new_df.to_csv(out_path + str(user), sep='\t', header=None, index=None)
            y_axis = new_df['distance'].tolist()
            draw_scatter_graph(user, 'Community Members', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + 0.1), os.path.join(out_path, user))

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

def calculate_aggregated_community_distances(median, internal, community):
    # calculates and stores the median JSD for each community into a file

    distance_dir = os.path.join(community, 'calculated_distances/')
    if internal:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon')
        if median:
            out_file = os.path.join(distance_dir, 'median_community_distances')
        else:
            out_file = os.path.join(distance_dir, 'mean_community_distances')
    else:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon_ext')
        if median:
            out_file = os.path.join(distance_dir, 'median_ext_community_distances')
        else:
            out_file = os.path.join(distance_dir, 'mean_ext_community_distances')

    if not os.path.exists(jsd_dists): return
    if os.path.exists(out_file): os.remove(out_file)

    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    df = pd.read_csv(jsd_dists, sep='\t', header=None, names=['user_1', 'user_2', 'distance'])
    if median:
        rows = ['jensen_shannon', np.median(df['distance'].tolist())]
        pd.DataFrame([rows]).to_csv(out_file, sep='\t', header=None, index=None)
    else:
        rows = ['jensen_shannon', np.mean(df['distance'].tolist())]
        pd.DataFrame([rows]).to_csv(out_file, sep='\t', header=None, index=None)

def calculate_external_distances(community):
    # creates a serialized dictionary containing each user in the community comparing the jensen shannon divergences
    # against randomly selected users from outside communities.

    # x-axis: users outside of community, y-axis: distance from observed user

    distance_dir = os.path.join(community, 'calculated_distances/')
    if(os.path.exists(os.path.join(distance_dir, 'median_external_community_distances'))): return
    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    
    if(len(comm_doc_vecs) <= 1): return
    
    distance_dir = os.path.join(community, 'calculated_distances/')
    jen_shan_file = os.path.join(distance_dir, 'jensen_shannon_ext')
    
    if os.path.exists(jen_shan_file): os.remove(jen_shan_file)

    community_pos = len(community.strip('/').split('/')) - 1
    working_dir = community.strip('/').split('/')[0:community_pos]
    working_dir = '/'.join(working_dir)

    with open(os.path.join(working_dir, 'document_vectors.json'), 'r') as all_community_doc_vecs_file:
        all_community_doc_vecs = json.load(all_community_doc_vecs_file)

    NUM_ITER = 10
    x_axis = np.arange(1, len(comm_doc_vecs))
    external_users = []

    with open(jen_shan_file, 'w') as out:
        for user in comm_doc_vecs:
            if not(external_users):
                external_users = get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER)
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
            for div in jsd:
                out.write('{}\t{}\t{}\n'.format(user, 'rand_user', div/NUM_ITER))

def get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER):
    # returns a multiple of a list of random users not in the current users' community

    # if number of iterations is set to 10, the random users returned is equal to:
    #  10 * (len(users in the community) - 1)

    # this calculation allows for a broader comparison of external users.

    max_external_users = len(all_community_doc_vecs) - len(comm_doc_vecs)
    internal_users = set(user for user in comm_doc_vecs)
    external_users = []
    while True:
        rand_external_user = random.sample(list(all_community_doc_vecs), 1)[0]
        if rand_external_user not in set(external_users) and rand_external_user not in internal_users:
            external_users.append(rand_external_user)
        if(len(external_users) == (len(comm_doc_vecs) - 1) * NUM_ITER or len(external_users) == max_external_users):
            return external_users

def user_distance_difference_graphs(community):
    # creates graph showing twitter user topic probabiliy distribution compared to a user of the same community
    # and also shows that same user compared against a user external to the community
    # all plotted data is separated into community directories

    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    
    if(len(comm_doc_vecs) <= 2): return

    out_path = os.path.join(community, 'distance_difference_graphs/jensen_shannon/')

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), 0o755)

    int_dists = os.path.join(community, 'calculated_distances/jensen_shannon')
    ext_dists = os.path.join(community, 'calculated_distances/jensen_shannon_ext')
    int_df = pd.read_csv(int_dists, sep='\t', header=None, names=['user_a', 'user_b', 'distance'])
    ext_df = pd.read_csv(ext_dists, sep='\t', header=None, names=['user_a', 'user_b', 'distance'])
    
    for user in comm_doc_vecs:
        try:
            df = int_df[(int_df.user_a == int(user)) | (int_df.user_b == int(user))]
            y_axis = df['distance'].tolist()
            plt.plot(np.arange(0, len(y_axis)), y_axis, 'b')
            df = ext_df[ext_df.user_a == int(user)]
            y_axis = df['distance'].tolist()
            plt.plot(np.arange(0, len(y_axis)), y_axis, 'g')
            plt.ylabel('Divergence')
            plt.title('Divergence from ' + user + ' to Internal & External Users')
            plt.ylim([0, np.log(2)])
            plt.xlabel('Users')
            plt.xlim([0, len(y_axis) - 1])
            plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
            plt.locator_params(axis ='x', nbins=len(y_axis) - 1)
            plt.locator_params(axis ='y', nbins=10)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(out_path + user)
            plt.close()
        
        except Exception as e:
            pass

def draw_dual_line_graph(title, x_label, y_label, y_axis_1, y_axis_2, line_1_label, line_2_label, output_path):
    x_axis = np.arange(0, len(y_axis_1))
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x_axis, y_axis_1, 'b')
    ax.plot(x_axis, y_axis_2, 'g', alpha=0.7)
    ax.legend([line_1_label, line_2_label], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.axis([0, np.amax(x_axis), 0, np.log(2) + .001])
    plt.margins(0.2)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    #plt.savefig(output_path + '.eps', format='eps')
    plt.savefig(output_path)
    plt.close(fig)

def user_topic_distribution_graph(community):
    # creates a graph of each user's LDA topic distribution from the generated model

    output_path = os.path.join(community, 'topic_distribution_graphs/')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), 0o755)

    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    for user in comm_doc_vecs:
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

def delete_small_communities(smallest_size, community):
    # if a clique has 'smallest_size' or less active members then the community and the clique will be
    # removed from the dataset
    vec_file = os.path.join(community, 'community_doc_vecs.json')
    comm_doc_vecs = open_community_document_vectors_file(vec_file)
    
    if len(comm_doc_vecs) <= smallest_size:
        shutil.rmtree(community)

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
            vec_file = os.path.join(os.path.join(path, community), 'community_doc_vecs.json')
            if os.path.exists(vec_file):
                yield(os.path.join(path, community))
        break

def calc_individual_dists_helper(status_file, internal, pool, total_work, working_dir):
    if internal:
        print('Calculating internal distances')
        for _ in tqdm.tqdm(pool.imap_unordered(calculate_internal_distances, dir_to_iter(working_dir)), total=total_work): pass
        with open(status_file, 'w') as out:
            out.write('')
    else:
        print('Calculating external distances')
        for _ in tqdm.tqdm(pool.imap_unordered(calculate_external_distances, dir_to_iter(working_dir)), total=total_work): pass
        with open(status_file, 'w') as out:
            out.write('')

def build_aggregated_dataframe(median, internal, community):
    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, 'community_doc_vecs.json'))
    
    if(len(comm_doc_vecs) <= 1): return
    
    comm_name = ntpath.basename(community)

    metric = 'median' if median else 'mean'
    pos = '' if internal else '_ext'
    dist_path = os.path.join(community, 'calculated_distances/' + metric + pos + '_community_distances')

    if os.path.exists(dist_path):
        df = pd.read_csv(dist_path, sep='\t', header=None, names=['metric', 'distance'])
        return [comm_name, len(comm_doc_vecs), float(df.distance)]

def main():
    parser = argparse.ArgumentParser(description="""Calculate median distances of users in communities and use those distances to plot graphs
                                                    to run all without disturbing the dataset:
                                                      python plot_distances.py -w working_dir/ -iedtms
                                                 """)

    subparsers = parser.add_subparsers(dest='mode')
    clean_parser = subparsers.add_parser('clean', help="""Remove inactive users and communities from dataset. Option to restore original dataset.
                                                          The working directory for the clean option should be where the active_users.json and
                                                          inactive_users.json file is located. That is usually in the folder that the downloaded
                                                          tweets are in.""")
    clean_parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='This is the directory with community graph and distance data')
    clean_parser.add_argument('-f', '--tweets_dir', required=True, action='store', dest='tweets_dir', help='This is the directory where tweets were downloaded to')
    clean_parser.add_argument('-o', action='store_true', required=True, help='Omit inactive communities from calculations/graphs')
    clean_parser.add_argument('-z', '--comm_size', nargs='?', type=int, const=1, default=2, help='Minimum size a clique or community should be to stay in dataset. Default = 2')

    indiv_parser = subparsers.add_parser('indiv', help='Caculate and graph individual internal or external user distances')
    indiv_parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='This is the directory to output data to')
    indiv_group = indiv_parser.add_mutually_exclusive_group(required=True)
    indiv_group.add_argument('-i', action='store_true', help='Calculate individual internal distances')
    indiv_group.add_argument('-e', action='store_true', help='Calculate individual external distances')
    indiv_group.add_argument('-I', action='store_true', help='Draw internal distance graphs')
    indiv_group.add_argument('-E', action='store_true', help='Draw external distance graphs')
    indiv_group.add_argument('-d', action='store_true', help='Draw internal vs external distance graphs') 

    collective_parser = subparsers.add_parser('collctv', help='Caculate and graph collective internal or external distances')
    collective_parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='This is the directory to output data to')
    collective_parser.add_argument('-m', '--median', action='store_true', dest='median', help='Use this option to set aggregation metric to median, default is average')
    collective_group = collective_parser.add_mutually_exclusive_group(required=True)
    collective_group.add_argument('-n', action='store_true', help='Calculate aggregated internal distances for communities')
    collective_group.add_argument('-x', action='store_true', help='Calculate aggregated external distances for communities')
    collective_group.add_argument('-a', action='store_true', help='Draw aggregated distance graphs for all communities')

    extended_parser = subparsers.add_parser('xtnded', help='Extra (experimental) distance calculations and graphs')
    extended_parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='This is the directory to output data to')
    extended_parser.add_argument('-t', action='store_true', help='Draw topic distribution graphs')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    print('Calculating total work required')
    total_work = len([community for community in tqdm.tqdm(dir_to_iter(args.working_dir))])
    
    int_status = os.path.join(args.working_dir, 'int_dists_status')
    ext_status = os.path.join(args.working_dir, 'ext_dists_status')

    if args.mode == 'clean':
        if args.o:
            func = partial(delete_small_communities, args.comm_size)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass

    if args.mode == 'indiv':
        if args.i:
            if not os.path.exists(int_status): calc_individual_dists_helper(int_status, True, pool, total_work, args.working_dir)
        if args.I:
            if os.path.exists(int_status):
                print('Drawing internal distance graphs')
                func = partial(individual_user_distance_graphs, True)
                for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            else:
                print('Error, cannot draw internal distance graphs; Calculate iexternal distances first')
        
        if args.e:
            if not os.path.exists(ext_status): calc_individual_dists_helper(ext_status, False, pool, total_work, args.working_dir)
        if args.E:
            if os.path.exists(ext_status):
                print('Drawing external distance graphs')
                func = partial(individual_user_distance_graphs, False)
                for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            else:
                print('Error, cannot draw external distance graphs; Calculate iexternal distances first')

        if args.d:
            if os.path.exists(int_status) and os.path.exists(ext_status):
                print('Drawing user internal vs external distance graphs')
                for _ in tqdm.tqdm(pool.imap_unordered(user_distance_difference_graphs, dir_to_iter(args.working_dir)), total=total_work): pass
            else:
                if not os.path.exists(int_status):
                    print('Error, cannot draw internal vs external distance graphs; Calculate internal distances first')
                if not os.path.exists(ext_status):
                    print('Error, cannot draw internal vs external distance graphs; Calculate external distances first')
                else:
                    print('Error, cannot draw internal vs external distance graphs; Calculate internal and external distances first')
    
    if args.mode == 'collctv':
        if args.n:
            print('Calculating internal aggregated community distances')
            func = partial(calculate_aggregated_community_distances, args.median, True)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            print('Building dataframe...')
            func = partial(build_aggregated_dataframe, args.median, True)
            result = pool.map(func, dir_to_iter(args.working_dir))
            result = [i for i in result if i]
            metric = 'median' if args.median else 'mean'
            pd.DataFrame(result).to_csv(os.path.join(args.working_dir, metric + '_aggregated_community_distances'), sep='\t', header=None, index=None)
        
        if args.x:
            print('Calculating external aggregated community distances')
            func = partial(calculate_aggregated_community_distances, args.median, False)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            print('Building dataframe...')
            func = partial(build_aggregated_dataframe, args.median, False)
            result = pool.map(func, dir_to_iter(args.working_dir))
            result = [i for i in result if i]
            metric = 'median' if args.median else 'mean'
            pd.DataFrame(result).to_csv(os.path.join(args.working_dir, metric + '_ext_aggregated_community_distances'), sep='\t', header=None, index=None)

        if args.a:
            metric = 'median' if args.median else 'mean'
            int_df = pd.read_csv(os.path.join(args.working_dir, metric + '_aggregated_community_distances'), sep='\t', header=None, names=['cid', 'size', 'dist'], index_col=0)
            ext_df = pd.read_csv(os.path.join(args.working_dir, metric + '_ext_aggregated_community_distances'), sep='\t', header=None, names=['cid', 'size', 'dist'], index_col=0)
            
            if int_df.index.str.contains('clique').any():
                int_y_axis = []
                ext_y_axis = []

                int_y_axis = int_df['dist'].tolist()
                ext_y_axis = ext_df['dist'].tolist()
    
                output_path = os.path.join(args.working_dir, metric + '_internal_external_divergence_graph')
                draw_dual_line_graph(metric.title() + ' Internal & External Divergence\n', 'Communities',
                                     ' Jensen Shannon Divergence', int_y_axis, ext_y_axis,
                                     'Internal', 'External', output_path)

    if args.mode == 'xtnded':
        if args.t:
            print('Drawing topic distribution graphs for all users')
            for _ in tqdm.tqdm(pool.imap_unordered(user_topic_distribution_graph, dir_to_iter(args.working_dir)), total=total_work): pass

    pool.terminate()

if __name__ == '__main__':
    sys.exit(main())
