import random
import pandas as pd
import json
import os
import ast
import sys
import argparse
import argcomplete
import scipy
import tqdm
from shutil import copyfile
from scipy.linalg import norm
from scipy.stats import entropy
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models, matutils

def calculate_internal_distances(community):
    '''
        for each user find the distance from every other user using their probability distribution vectors

        Dictionary <k, v>(user_id, distribution_vector)

    '''
    distance_dir = os.path.join(community, 'calculated_distances/')
    if(os.path.exists(os.path.join(distance_dir, 'median_community_distances'))): return 
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return
    if not os.path.exists(os.path.dirname(distance_dir)):
        os.makedirs(os.path.dirname(distance_dir), 0o755)
    
    jen_shan_file = os.path.join(distance_dir, 'jensen_shannon')
    if os.path.exists(jen_shan_file): os.remove(jen_shan_file)
    with open(jen_shan_file, 'w') as out:
        for key in sorted(comm_doc_vecs):
            user = key
            # only necessary to compare each user with any other user once
            vec_1 = comm_doc_vecs.pop(key)

            for key_2 in sorted(comm_doc_vecs):
                vec_2 = comm_doc_vecs[key_2]
                out.write('{}\t{}\t{}\n'.format(user, key_2, jensen_shannon_divergence(vec_1, vec_2)))

def jensen_shannon_divergence(P, Q):
    _P = np.array(P) / norm(np.array(P), ord=1)
    _Q = np.array(Q) / norm(np.array(Q), ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def individual_user_distance_graphs(internal, community):
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences between
    their probability distribution vectors against other users 

    x-axis: users in community, y-axis: distance from observed user 
    '''
    distance_dir = os.path.join(community, 'calculated_distances/')
    if internal:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon')
        out_path = os.path.join(os.path.join(community, 'internal_user_graphs'), 'jensen_shannon')
        out_file = os.path.join(distance_dir, 'community_distances')
    else:
        jsd_dists = os.path.join(distance_dir, 'jensen_shannon_ext')
        out_path = os.path.join(os.path.join(community, 'external_user_graphs'), 'jensen_shannon')
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
            new_df.to_csv(jsd_path + str(user), sep='\t', header=None, index=None)
            y_axis = new_df['distance'].tolist()
            draw_scatter_graph(user, 'Community Members', 'Jensen Shannon Divergence', x_axis, y_axis, 0, len(x_axis) + 1, 0, (np.log(2) + 0.1), os.path.join(jsd_path, user))

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
    '''
    calculates and stores the median JSD for each community into a file
    
    '''
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
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences 
    against randomly selected users from outside communities.

    x-axis: users outside of community, y-axis: distance from observed user 

    '''
    distance_dir = os.path.join(community, 'calculated_distances/')
    if(os.path.exists(os.path.join(distance_dir, 'median_external_community_distances'))): return 
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return
    distance_dir = os.path.join(community, 'calculated_distances/')
    jen_shan_file = os.path.join(distance_dir, 'jensen_shannon_ext')
    if os.path.exists(jen_shan_file): os.remove(jen_shan_file)

    working_dir = community.strip('/').split('/')[0]
    with open(os.path.join(working_dir, 'document_vectors.json'), 'r') as all_community_doc_vecs_file:
        all_community_doc_vecs = json.load(all_community_doc_vecs_file)

    jsd_path = community + '/calculate_external_distancess/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

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

def user_distance_difference_graphs(community):
    '''
    user to internal against user to external distance
    graphs, puts plotted data into community directories

    '''
    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, '/community_doc_vecs.json'))
    if(len(comm_doc_vecs) <= 1): return

    jsd_path = os.path.join(community, '/distance_difference_graphs/jensen_shannon/')

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)
    
    int_dists = os.path.join(community, '/calculated_distances/jensen_shannon')
    ext_dists = os.path.join(community, '/calculated_distances/jensen_shannon_ext')
    int_df = pd.read_csv(int_dists, sep='\t', header=None, names=['user_a', 'user_b', 'distance'])
    ext_df = pd.read_csv(ext_dists, sep='\t', header=None, names=['user_a', 'user_b', 'distance'])
    for user in comm_doc_vecs:
        if not os.path.exists(os.path.join(jsd_path, user + '.png')):
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
            plt.locator_params(nbins=25)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(jsd_path + user)
            plt.savefig(jsd_path + user + '.eps', format='eps')
            plt.close()

def community_aggregated_int_ext_distance(median, working_dir):
    '''
    graphs displaying median internal vs median external distances for all the communities & cliques

    '''
    metric = 'Median' if median else 'Mean'
    uncap_metric = metric[0].lower() + metric[1:]
    int_clq_y_axis, int_comm_y_axis = community_aggregated_int_ext_distance_y_axes(working_dir, uncap_metric + '_community_distances')
    ext_clq_y_axis, ext_comm_y_axis = community_aggregated_int_ext_distance_y_axes(working_dir, uncap_metric + '_external_community_distances')

    if int_clq_y_axis and ext_clq_y_axis:
        output_path = os.path.join(working_dir, metric + '_clique_internal_external_divergence')
        draw_dual_line_graph(metric + ' Internal & External Community Divergence', 'Clique ID', 
                             metric + ' Jensen Shannon Divergence', int_clq_y_axis, ext_clq_y_axis,
                             'Internal', 'External', output_path)

    if int_comm_y_axis and ext_comm_y_axis:
        output_path = os.path.join(working_dir, metric + '_community_internal_external_divergence')
        draw_dual_line_graph(metric + ' Internal & External Community Divergence', 'Community ID', 
                             metric + ' Jensen Shannon Divergence', int_comm_y_axis, ext_comm_y_axis,
                             'Internal', 'External', output_path)

    if int_clq_y_axis and int_comm_y_axis:
        output_path = os.path.join(working_dir, metric + '_clique_community_internal_divergence')
        draw_dual_line_graph(metric + ' Internal & External Community Divergence', 'Clique-Community ID', 
                             metric + ' Jensen Shannon Divergence', int_clq_y_axis, int_comm_y_axis,
                             'Clique', 'Community', output_path)

    if ext_clq_y_axis and ext_comm_y_axis:
        output_path = os.path.join(working_dir, metric + '_clique_community_external_divergence')
        draw_dual_line_graph(metric + ' Internal & External Community Divergence', 'Clique-Community ID', 
                             metric + ' Jensen Shannon Divergence', ext_clq_y_axis, ext_comm_y_axis,
                             'Clique', 'Community', output_path)

def community_aggregated_int_ext_distance_y_axes(working_dir, filename):
    clq_y_axis = []
    comm_y_axis = []
    for path, dirs, files in os.walk(working_dir):
        for community in sorted(dirs):
            distance_dir = os.path.join(os.path.join(path, community), 'calculated_distances/')
            if os.path.exists(os.path.join(distance_dir, filename)):
                df = pd.read_csv(os.path.join(distance_dir, filename), sep='\t', header=None, names=['metric', 'distance'])
                if 'clique' in community:
                    clq_y_axis.append(float(df[df.metric == 'jensen_shannon']['distance']))
                else:
                    comm_y_axis.append(float(df[df.metric == 'jensen_shannon']['distance']))
        break # restrict depth of folder traversal to 1
    return clq_y_axis, comm_y_axis

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
    plt.savefig(output_path + '.eps', format='eps')
    plt.savefig(output_path)
    plt.close(fig)

def overall_int_dist_wrt_comm_size(median, working_dir):
    '''
    displays the overall median internal clique and internal community divergence in relation
    to the sizes of the cliques or communities

    ''' 
    clq_dists = defaultdict(list)
    comm_dists = defaultdict(list)
    metric = 'Median' if median else 'Mean'
    uncap_metric = metric[0].lower() + metric[1:]
    
    for path, dirs, files in os.walk(working_dir):
        for community in sorted(dirs):
            comm_path = os.path.join(path, community)
            comm_doc_vecs = open_community_document_vectors_file(os.path.join(comm_path, 'community_doc_vecs.json'))
            if(len(comm_doc_vecs) <= 1): continue
            dist_path = os.path.join(comm_path, 'calculated_distances/' + uncap_metric + '_community_distances')
            if os.path.exists(dist_path):
                df = pd.read_csv(dist_path, sep='\t', header=None, names=['metric', 'distance'])
                if 'clique' in community:
                    clq_dists[len(comm_doc_vecs)].append(float(df[df.metric == 'jensen_shannon']['distance']))
                else:
                    comm_dists[len(comm_doc_vecs)].append(float(df[df.metric == 'jensen_shannon']['distance']))
        break

    clq_x_axis = []
    clq_y_axis = []
    comm_x_axis = []
    comm_y_axis = []
    print('Drawing overall ' + uncap_metric + ' clique & community divergence by size graph')

    plt.ylabel(metric + ' Jensen Shannon Divergence\n')
    plt.xlabel('Size of Community/Clique')
    plt.title(metric + ' Community/Clique Similarity Distribution')

    for clq_size in clq_dists:
        clq_x_axis.append(clq_size)
        clq_y_axis.append(np.mean(clq_dists[clq_size]))

    for comm_size in comm_dists:
        comm_x_axis.append(comm_size)
        comm_y_axis.append(np.mean(comm_dists[comm_size]))

    output_path = os.path.join(working_dir, uncap_metric + '_overall_internal_distance_by_community_size')
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
    output_path = os.path.join(community, '/topic_distribution_graphs/')
            
    if not os.path.exists(os.path.dirname(output_path)):
    	os.makedirs(os.path.dirname(output_path), 0o755)

    comm_doc_vecs = open_community_document_vectors_file(os.path.join(community, '/community_doc_vecs.json'))
    for user in comm_doc_vecs:
        if not os.path.exists(os.path.join(output_path, user) + '.png'):
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

def restore_original_dataset(working_dir, community):
    bak_file = os.path.join(community, 'community_doc_vecs.json.bak')
    res_file = os.path.join(community, 'community_doc_vecs.json')
    if os.path.exists(bak_file):
        copyfile(bak_file, res_file)
        os.remove(bak_file)

def delete_inactive_communities(community):
    '''
    if a clique has 1 or less active members then the community and the clique must be
    removed from the dataset
    '''
    vec_file = os.path.join(community, 'community_doc_vecs.json')
    comm_vec_file = os.path.join(community.replace('clique', 'community'), 'community_doc_vecs.json')
    comm_doc_vecs = open_community_document_vectors_file(vec_file)
    if len(comm_doc_vecs) <= 1:
        if os.path.exists(vec_file):
            print('Removing: ' + str(community))
            os.remove(vec_file)
        if os.path.exists(comm_vec_file):
            print('Removing: ' + str(community.replace('clique', 'community')))
            os.remove(comm_vec_file)

def delete_inactive_users(community):
    '''
    removes users from the dataset if the amount of times they tweeted is less than 10

    '''
    bak_file = os.path.join(community, 'community_doc_vecs.json.bak')
    vec_file = os.path.join(community, 'community_doc_vecs.json')
    comm_doc_vecs = open_community_document_vectors_file(vec_file)
    if comm_doc_vecs:
        copyfile(vec_file, bak_file) 

    with open('dnld_tweets/inactive_users.json', 'r') as infile:
        inactive_users = json.load(infile)

    for user in inactive_users:
        if user in comm_doc_vecs:
            print("Removing inactive user: " + user)
            del comm_doc_vecs[user]
          
    with open(vec_file, 'w') as comm_doc_vecs_file:
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
    comm_name = community.strip('/').split('/')[1]
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
    parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='Location of working directory')
    parser.add_argument('-o', action='store_true', help='Omit inactive users and communities from calculations/graphs')
    parser.add_argument('-r', action='store_true', help='Revert to original dataset (use if removed inactive users and want to restore dataset)')
    parser.add_argument('-i', action='store_true', help='Calculate individual internal distances')
    parser.add_argument('-e', action='store_true', help='Calculate individual external distances')
    parser.add_argument('-I', action='store_true', help='Draw internal distance graphs')
    parser.add_argument('-E', action='store_true', help='Draw external distance graphs')
    parser.add_argument('-n', action='store_true', help='Calculate aggregated internal distances for communities')
    parser.add_argument('-x', action='store_true', help='Calculate aggregated external distances for communities')
    parser.add_argument('-m', '--median', action='store_true', dest='median', help='Use this option to set aggregation metric to median, default is average')
    parser.add_argument('-d', action='store_true', help='Draw internal vs external distance graphs')
    parser.add_argument('-t', action='store_true', help='Draw topic distribution graphs')
    parser.add_argument('-a', action='store_true', help='Draw aggregated distance graphs for communities')
    parser.add_argument('-s', action='store_true', help='Draw distance vs size distribution graphs for communities')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    print('Calculating total work required')
    total_work = len([community for community in tqdm.tqdm(dir_to_iter(args.working_dir))])
    int_status = os.path.join(args.working_dir, 'int_dists_status')
    ext_status = os.path.join(args.working_dir, 'ext_dists_status')
	
    if args.o:
        pool.map(delete_inactive_users, dir_to_iter(args.working_dir))
        pool.map(delete_inactive_communities, dir_to_iter(args.working_dir))
    if args.r:
        func = partial(restore_original_dataset, args.working_dir)
        pool.map(restore_original_dataset, dir_to_iter(args.working_dir))
    if args.i:
        if not os.path.exists(int_status): calc_individual_dists_helper(int_status, True, pool, total_work, args.working_dir)
    if args.I:
        if os.path.exists(int_status):
            print('Drawing internal distance graphs')
            func = partial(individual_user_distance_graphs, True)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
        else:
            print('Error, cannot draw internal distance graphs; Calculating internal distances first')
            calc_individual_dists_helper(int_status, True)
    if args.e:
        if not os.path.exists(ext_status): calc_individual_dists_helper(ext_status, False, pool, total_work, args.working_dir)
    if args.E:
        if os.path.exists(ext_status):
            print('Drawing external distance graphs')
            func = partial(individual_user_distance_graphs, False)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
        else:
            print('Error, cannot draw external distance graphs; Calculating external distances first')
            calc_individual_dists_helper(ext_status, False)
    if args.n:
        if os.path.exists(int_status):
            print('Calculating internal aggregated community distances')
            func = partial(calculate_aggregated_community_distances, args.median, True)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            print('Building dataframe...')
            func = partial(build_aggregated_dataframe, args.median, True)
            result = pool.map(func, dir_to_iter(args.working_dir))
            metric = 'median' if args.median else 'mean'
            pd.DataFrame(result).to_csv(os.path.join(args.working_dir, metric + '_aggregated_community_distances'), sep='\t', header=None, index=None)
        else:
            print('Error, cannot calculate aggregated community distances; Calculating internal distances first')
            calc_individual_dists_helper(int_status, True)
    if args.x:
        if os.path.exists(ext_status): 
            print('Calculating external aggregated community distances')
            func = partial(calculate_aggregated_community_distances, args.median, False)
            for _ in tqdm.tqdm(pool.imap_unordered(func, dir_to_iter(args.working_dir)), total=total_work): pass
            print('Building dataframe...')
            func = partial(build_aggregated_dataframe, args.median, False)
            result = pool.map(func, dir_to_iter(args.working_dir))
            metric = 'median' if args.median else 'mean'
            pd.DataFrame(result).to_csv(os.path.join(args.working_dir, metric + '_ext_aggregated_community_distances'), sep='\t', header=None, index=None)
        else:
            print('Error, cannot calculate aggregated community distances; Calculating external distances first')
            calc_individual_dists_helper(ext_status, False)
    if args.d:
        if os.path.exists(int_status) and os.path.exists(ext_status):
            print('Drawing user internal vs external distance graphs')
            for _ in tqdm.tqdm(pool.imap_unordered(user_distance_difference_graphs, dir_to_iter(args.working_dir)), total=total_work): pass
        else:
            if not os.path.exists(int_status):
                print('Error, cannot draw internal vs external distance graphs; Calculating internal distances first')
                calc_individual_dists_helper(int_status, True)
            if not os.path.exists(ext_status):
                print('Error, cannot draw internal vs external distance graphs; Calculating external distances first')
                calc_individual_dists_helper(ext_status, False)
            else:
                print('Error, cannot draw internal vs external distance graphs; Calculating internal and external distances first')
                calc_individual_dists_helper(int_status, True)
                calc_individual_dists_helper(ext_status, False)
    if args.t:
        print('Drawing topic distribution graphs for all users')
        for _ in tqdm.tqdm(pool.imap_unordered(user_topic_distribution_graph, dir_to_iter(args.working_dir)), total=total_work): pass

    pool.terminate()
 
    if args.a:
        metric = 'Median' if args.median else 'Mean'
        uncap_metric = metric[0].lower() + metric[1:]
        int_df = pd.read_csv(os.path.join(args.working_dir, uncap_metric + '_aggregated_community_distances'), sep='\t', header=None, names=['cid', 'size', 'dist'], index_col=0)
        ext_df = pd.read_csv(os.path.join(args.working_dir, uncap_metric + '_ext_aggregated_community_distances'), sep='\t', header=None, names=['cid', 'size', 'dist'], index_col=0)
        if int_df.index.str.contains('clique').any():
            int_clq_y_axis = []
            int_clq_y_axis1 = []
            ext_clq_y_axis = []
            ext_clq_y_axis1 = []
            

#            int_clq_df = int_df[int_df.index.str.contains('clique')]
#            int_clq_y_axis = int_clq_df['dist'][(int_clq_df['size'] > 3) & (int_clq_df['size'] <= 150)].tolist()
#            int_clq_y_axis1 = int_clq_df['dist'][int_clq_df['size'] > 150].tolist()
#            if not int_clq_y_axis: print('int_clq_y_axis > 3 <=150')
#            if not int_clq_y_axis1: print('int_clq_y_axis > 150')
#        
#            ext_clq_df = ext_df[ext_df.index.str.contains('clique')]
#            ext_clq_y_axis = ext_clq_df['dist'][(ext_clq_df['size'] > 3) & (ext_clq_df['size'] <= 150)].tolist()
#            ext_clq_y_axis1 = ext_clq_df['dist'][ext_clq_df['size'] > 150].tolist()
#            if not ext_clq_y_axis: print('ext_clq_y_axis > 3 <=150')
#            if not ext_clq_y_axis1: print('ext_clq_y_axis > 150')
            
            int_comm_df = int_df[int_df.index.str.contains('community')]
            int_comm_y_axis = int_comm_df['dist'][(int_comm_df['size'] > 3) & (int_comm_df['size'] <= 150)].tolist()
            int_comm_y_axis1 = int_comm_df['dist'][int_comm_df['size'] > 150].tolist()
            if not int_comm_y_axis: print('int_comm_y_axis > 3 <=150')
            if not int_comm_y_axis1: print('int_comm_y_axis > 150')
        
            ext_comm_df = ext_df[ext_df.index.str.contains('community')]
            ext_comm_y_axis = ext_comm_df['dist'][(ext_comm_df['size'] > 3) & (ext_comm_df['size'] <= 150)].tolist()
            ext_comm_y_axis1 = ext_comm_df['dist'][ext_comm_df['size'] > 150].tolist()
            if not ext_comm_y_axis: print('ext_comm_y_axis > 3 <=150')
            if not ext_comm_y_axis1: print('ext_comm_y_axis > 150')

            int_comm_y_axis = int_df['dist'][int_df.index.str.contains('community')].tolist()
            int_clq_y_axis = int_df['dist'][int_df.index.str.contains('clique')].tolist()
            ext_comm_y_axis = ext_df['dist'][ext_df.index.str.contains('community')].tolist()
            ext_clq_y_axis = ext_df['dist'][ext_df.index.str.contains('clique')].tolist()

            if int_clq_y_axis and ext_clq_y_axis:
                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_internal_external_divergence_CAA')
                draw_dual_line_graph(metric + ' Internal & External Divergence\n for Cliques Using CAA', 'Cliques', 
                                     ' Jensen Shannon Divergence', int_clq_y_axis, ext_clq_y_axis,
                                     'Internal', 'External', output_path)
            
#            if int_clq_y_axis1 and ext_clq_y_axis1:
#                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_internal_external_divergence_gt150_CAA')
#                draw_dual_line_graph(metric + ' Internal & External Divergence\n for Cliques of Size > 150 Using CAA', 'Cliques', 
#                                     metric + ' Jensen Shannon Divergence', int_clq_y_axis1, ext_clq_y_axis1,
#                                     'Internal', 'External', output_path)
#
            if int_comm_y_axis and ext_comm_y_axis:
                output_path = os.path.join(args.working_dir, uncap_metric + '_community_internal_external_divergence_CAA')
                draw_dual_line_graph(metric + ' Internal & External Divergence for\n Communities Using CAA', 'Communities', 
                                     ' Jensen Shannon Divergence', int_comm_y_axis, ext_comm_y_axis,
                                     'Internal', 'External', output_path)
#
#            if int_comm_y_axis1 and ext_comm_y_axis1:
#                output_path = os.path.join(args.working_dir, uncap_metric + '_community_internal_external_divergence_gt150_CAA')
#                draw_dual_line_graph(metric + ' Internal & External Divergence for\n Communities of Size > 150 Using CAA', 'Communities', 
#                                     metric + ' Jensen Shannon Divergence', int_comm_y_axis1, ext_comm_y_axis1,
#                                     'Internal', 'External', output_path)

            if int_clq_y_axis and int_comm_y_axis:
                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_community_internal_divergence_CAA')
                draw_dual_line_graph(metric + ' Internal Divergence for\n Cliques & Communties Using CAA', 'Cliques/Communities', 
                                     ' Jensen Shannon Divergence', int_clq_y_axis, int_comm_y_axis,
                                     'Clique', 'Community', output_path)

#            if int_clq_y_axis1 and int_comm_y_axis1:
#                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_community_internal_divergence_gt150_CAA')
#                draw_dual_line_graph(metric + ' Internal Divergence for Cliques & Communities\n of Size > 150 Using CAA', 'Cliques/Communities', 
#                                     metric + ' Jensen Shannon Divergence', int_clq_y_axis1, int_comm_y_axis1,
#                                     'Clique', 'Community', output_path)
#
#            if ext_clq_y_axis and ext_comm_y_axis:
#                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_community_external_divergence_gt3lte150_CAA')
#                draw_dual_line_graph(metric + ' External Divergence for Cliques & Communties\nof Size > 3 & <= 150 Using CAA', 'Cliques/Communities', 
#                                     metric + ' Jensen Shannon Divergence', ext_clq_y_axis, ext_comm_y_axis,
#                                     'Clique', 'Community', output_path)
#
#            if ext_clq_y_axis1 and ext_comm_y_axis1:
#                output_path = os.path.join(args.working_dir, uncap_metric + '_clique_community_external_divergence')
#                draw_dual_line_graph(metric + ' External Divergence for Cliques & Communities\n of Size > 150 Using CAA', 'Cliques/Communities', 
#                                     metric + ' Jensen Shannon Divergence', ext_clq_y_axis1, ext_comm_y_axis1,
#                                     'Clique', 'Community', output_path)

        else:
            int_comm_y_axis = int_df['dist'].tolist()
            ext_comm_y_axis = ext_df['dist'].tolist()
            
            output_path = os.path.join(args.working_dir, uncap_metric + '_community_internal_external_divergence_InfoMap')
            draw_dual_line_graph(metric + ' Internal & External Community Divergence\n for Communities Using InfoMap', 'Communities', 
                                 ' Jensen Shannon Divergence', int_comm_y_axis, ext_comm_y_axis,
                                 'Internal', 'External', output_path)
#            int_comm_y_axis = int_df['dist'][(int_df['size'] > 3) & (int_df['size'] <= 150)].tolist()
#            int_comm_y_axis1 = int_df['dist'][int_df['size'] > 150].tolist()
#            ext_comm_y_axis = ext_df['dist'][(ext_df['size'] > 3) & (ext_df['size'] <= 150)].tolist()
#            ext_comm_y_axis1 = ext_df['dist'][ext_df['size'] > 150].tolist()
#
#            output_path = os.path.join(args.working_dir, uncap_metric + '_community_internal_external_divergence_lte150_InfoMap')
#            draw_dual_line_graph(metric + ' Internal & External Community Divergence\n for Communities of Size > 3 & <=150 Using InfoMap', 'Communities', 
#                                 ' Jensen Shannon Divergence', int_comm_y_axis, ext_comm_y_axis,
#                                 'Internal', 'External', output_path)
#
#            output_path = os.path.join(args.working_dir, uncap_metric + '_community_internal_external_divergence_gt150_InfoMap')
#            draw_dual_line_graph(metric + ' Internal & External Community Divergence\n for Communities of Size > 150 Using InfoMap', 'Communities', 
#                                 ' Jensen Shannon Divergence', int_comm_y_axis1, ext_comm_y_axis1,
#                                 'Internal', 'External', output_path)
    if args.s:
        overall_int_dist_wrt_comm_size(args.working_dir)

if __name__ == '__main__':
    sys.exit(main())

