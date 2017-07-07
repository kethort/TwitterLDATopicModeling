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
import MySQLdb
from shutil import copyfile
from scipy.spatial import distance
from scipy.linalg import norm
from scipy.stats import entropy
import multiprocessing
from functools import partial
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict
from collections import OrderedDict
from gensim import corpora, models, matutils

def insert_into_db(table_name, comm_name, working_dir, attr):
    conn = MySQLdb.connect(host='localhost', user='root', passwd='yetanotherpassword', db='distances')
    c = conn.cursor()
    c.execute('SET sql_notes=0')
    c.execute('CREATE TABLE IF NOT EXISTS {tn} ({idx} {idxt}, {f0} {f0t}, {f1} {f1t},\
               {f2} {f2t}, {f3} {f3t}, {f4} {f4t}, {f5} {f5t}, {f6} {f6t})'\
               .format(tn=table_name, idx='uid', idxt='INTEGER PRIMARY KEY NOT NULL AUTO_INCREMENT', f0='cid', f0t='TEXT',
                       f1='user_a', f1t='TEXT', f2='user_b', f2t='TEXT', f3='cos', f3t='REAL', f4='euc', 
                       f4t='REAL', f5='hel', f5t='REAL', f6='jen', f6t='REAL'))
    c.execute('SET sql_notes=1')
    try:
        if table_name == 'internal_distances':
            c.execute("""INSERT INTO internal_distances (cid, user_a, user_b, cos, euc, hel, jen) VALUES (%s, %s, %s, %s, %s, %s, %s)""", (comm_name, attr[0], attr[1], attr[2], attr[3], attr[4], attr[5]))
        if table_name == 'external_distances':
            c.execute("""INSERT INTO external_distances (cid, user_a, user_b, cos, euc, hel, jen) VALUES (%s, %s, %s, %s, %s, %s, %s)""", (comm_name, attr[0], attr[1], attr[2], attr[3], attr[4], attr[5]))

    except Exception as e:
        print('Database insert exception {}'.format(e)) 
    conn.commit()
    conn.close()

def internal_distances_to_iter(comm_doc_vecs): 
    for user_1 in sorted(comm_doc_vecs):
        vec_1 = comm_doc_vecs.pop(user_1)
        for user_2 in sorted(comm_doc_vecs):
            vec_2 = comm_doc_vecs[user_2]
            yield [user_1, user_2, distance.cosine(vec_1, vec_2), distance.euclidean(vec_1, vec_2),
                   hellinger_distance(vec_1, vec_2), jensen_shannon_divergence(vec_1, vec_2)]

def calculate_internal_distances(community):
    '''
        for each user find the distance from every other user using their probability distribution vectors
        n(n-1)/2 transactions
    '''
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return
    distances = []
    working_dir = community.strip('/').split('/')[0]
    comm_name = community.strip('/').split('/')[1]

    func = partial(insert_into_db, 'internal_distances', comm_name, working_dir)
    pool.map(func, internal_distances_to_iter(comm_doc_vecs))
    pool.close()

def hellinger_distance(P, Q):
    return distance.euclidean(np.sqrt(np.array(P)), np.sqrt(np.array(Q))) / np.sqrt(2)

def jensen_shannon_divergence(P, Q):
    _P = np.array(P) / norm(np.array(P), ord=1)
    _Q = np.array(Q) / norm(np.array(Q), ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def user_distance_graphs(df, measure, ovrwrt, community):
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences between
    their probability distribution vectors against other users in same community

    x-axis: users in community, y-axis: distance from observed user 
    '''
    jsd_path = os.path.join(community, measure + '_user_distance_graphs/jensen_shannon/')
    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return

    working_dir = community.strip('/').split('/')[0]
    comm_name = community.strip('/').split('/')[1]
    print('Drawing {} distance graphs for community: {}'.format(measure, comm_name))
    x_axis = np.arange(1, len(comm_doc_vecs))
    df = df.loc[df['cid'] == comm_name]
    for user in comm_doc_vecs:
        if not os.path.exists(jsd_path + user + '.png') or ovrwrt:
            new_df = df[(df.user_a == user) | (df.user_b == user)]
            y_axis = new_df['jen'].tolist()
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

def external_distances_to_iter(comm_doc_vecs, all_community_doc_vecs):
    external_users = []
    for user in comm_doc_vecs:
        if not(external_users):
            external_users = get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER)
        i = 0
        cos = 0
        euc = 0
        hel = 0
        jen = 0
        while(i < (len(comm_doc_vecs) - 1) * NUM_ITER):
            rand_user = external_users.pop()
            external_users.insert(0, rand_user)
            cos += distance.cosine(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
            euc += distance.euclidean(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
            hel += hellinger_distance(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
            jen += jensen_shannon_divergence(all_community_doc_vecs[user], all_community_doc_vecs[rand_user])
            i += 1
        yield [user, 'rand_user', cos/NUM_ITER, euc/NUM_ITER, hel/NUM_ITER, jen/NUM_ITER]

def calculate_external_distances(community):
    '''
    creates graph displaying each user in the community comparing the jensen shannon divergences 
    against randomly selected users from outside communities.

    NUM_ITER is a variable for smoothing the sample by calculating the median of the distances 
    between a user and n random users equal in size to the user's community over NUM_ITER
    times. The result is then normalized back to distance metric constraints. 
    **set NUM_ITER to 1 to ignore this process

    x-axis: users outside of community, y-axis: distance from observed user 

    '''
    NUM_ITER = 5 
    working_dir = community.strip('/').split('/')[0]
    comm_name = '/'.join(community.strip('/').split('/')[1:])
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return
    all_community_doc_vecs = open_community_document_vectors_file(os.path.join(working_dir, 'document_vectors.json'))
    func = partial(insert_into_db, 'external_distances', comm_name, working_dir)
    pool.map(func, external_distances_to_iter(comm_doc_vecs))
    pool.close()

def get_rand_users(all_community_doc_vecs, comm_doc_vecs, NUM_ITER):
    '''
    returns multiple of a list of random users not in the current users' community

    if number of iterations is set to 10, the random users returned is equal to:
      10 * (len(users in the community) - 1)

    '''
    internal_users = set(user for user in comm_doc_vecs)
    external_users = set(user for user in all_community_doc_vecs) - internal_users
    if(len(comm_doc_vecs) * NUM_ITER > len(external_users)):
        return list(external_users)
    else:
        return list(random.sample(external_users, len(comm_doc_vecs) * NUM_ITER))

def user_internal_external_graphs(ovrwrt, community):
    '''
    user to internal against user to external distance
    graphs, puts plotted data into community directories

    '''
    working_dir = community.strip('/').split('/')[0]
    comm_name = '/'.join(community.strip('/').split('/')[1:])
    columns=['cid', 'user_a', 'user_b', 'cos', 'euc', 'hel', 'jen']
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if(len(comm_doc_vecs) <= 1): return

    jsd_path = community + '/distance_difference_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    print('Drawing internal vs external distance for: ' + str(community))
    for user in comm_doc_vecs:
        if not os.path.exists(jsd_path + user + '.png') or ovrwrt:
            df = pd.read_csv(os.path.join(working_dir, 'internal_distances'), sep='\t')
            df = df.loc[df['cid'] == comm_name]
            int_df = df[(df.user_a == int(user)) | (df.user_b == int(user))]
            y_axis = int_df['jen'].tolist()
            plt.plot(np.arange(0, len(y_axis)), y_axis, 'b')

            df = pd.read_csv(os.path.join(working_dir, 'external_distances'), sep='\t')
            df = df.loc[df['cid'] == comm_name]
            ext_df = df[df.user_a == int(user)]
            y_axis = ext_df['jen'].tolist()
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
            plt.close()

def community_aggregated_internal_external_distance_graphs(working_dir):
    '''
    graphs displaying median internal vs median external distances for all the communities & cliques

    '''
    int_clq_y_axis, int_comm_y_axis = community_aggregated_internal_external_distance_graphs_y_axes(os.path.join(working_dir, 'internal_aggregated_distances'))
    ext_clq_y_axis, ext_comm_y_axis = community_aggregated_internal_external_distance_graphs_y_axes(os.path.join(working_dir, 'external_aggregated_distances'))

    output_path = working_dir + 'median_clique_internal_external_divergence'
    draw_dual_line_graph('Median Internal & External Clique Divergence', 'Clique ID', 
                         'Median Jensen Shannon Divergence', int_clq_y_axis, ext_clq_y_axis,
                         'Internal', 'External', output_path)

    output_path = working_dir + 'median_community_internal_external_divergence'
    draw_dual_line_graph('Median Internal & External Community Divergence', 'Community ID', 
                         'Median Jensen Shannon Divergence', int_comm_y_axis, ext_comm_y_axis,
                         'Internal', 'External', output_path)

    output_path = working_dir + 'median_clique_community_internal_divergence'
    draw_dual_line_graph('Median Internal Clique/Community Divergence', 'Clique-Community ID', 
                         'Median Jensen Shannon Divergence', int_clq_y_axis, int_comm_y_axis,
                         'Clique', 'Community', output_path)

    output_path = working_dir + 'median_clique_community_external_divergence'
    draw_dual_line_graph('Median External Clique/Community Divergence', 'Clique-Community ID', 
                         'Median Jensen Shannon Divergence', ext_clq_y_axis, ext_comm_y_axis,
                         'Clique', 'Community', output_path)

def community_aggregated_internal_external_distance_graphs_y_axes(filename):
    columns = ['cos_med', 'cos_mean', 'euc_med', 'euc_mean', 'hel_med', 'hel_mean', 'jen_med', 'jen_mean']
    df = pd.read_csv(filename, sep='\t')
    clq_y_axis = df[df['cid'].str.contains('clique')]['jen_med'].tolist()
    comm_y_axis = df[df['cid'].str.contains('community')]['jen_med'].tolist()
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
    plt.savefig(output_path)
    plt.close(fig)

def aggregated_internal_distance_by_community_size_graph(working_dir):
    '''
    displays the overall median internal clique and internal community divergence in relation
    to the sizes of the cliques or communities

    ''' 
    agg_cols = ['cos_med', 'cos_mean', 'euc_med', 'euc_mean', 'hel_med', 'hel_mean', 'jen_med', 'jen_mean', 'size']
    agg_metrics = OrderedDict([('cos',['median', 'mean']), ('euc', ['median', 'mean']), ('hel', ['median', 'mean']), ('jen', ['median', 'mean']), ('cid', 'size')])
    dub_agg_metrics = OrderedDict([('cos_med', 'median'), ('cos_mean', 'mean'), ('euc_med', 'median'), ('euc_mean', 'mean'), ('hel_med', 'median'), ('hel_mean', 'mean'), ('jen_med', 'median'), ('jen_mean', 'mean')])
    df = pd.read_csv(os.path.join(working_dir, 'internal_distances'), sep='\t')
    clq_df = df[df['cid'].str.contains('clique')].groupby(['cid']).agg(agg_metrics)
    clq_df.columns = agg_cols
    clq_df = clq_df.reset_index().groupby(['size']).agg(dub_agg_metrics).reset_index()
    
    comm_df = df[df['cid'].str.contains('community')].groupby(['cid']).agg(agg_metrics)
    comm_df.columns = agg_cols
    comm_df = comm_df.reset_index().groupby(['size']).agg(dub_agg_metrics).reset_index()
    
    ax = clq_df.plot(kind='scatter', marker='x', color='g', x='size', y='jen_med')
    comm_df.plot(kind='scatter', color='b', x='size', y='jen_med', ax=ax, 
                 ylim=(0, np.log(2) + .001), xlim=(-5, comm_df['size'].max() + 5),
                 title='Median Community/Clique Similarity by Size')
    print('Drawing overall median clique & community divergence by size graph')
    plt.ylabel('Median Jensen Shannon Divergence\n')
    plt.xlabel('Size of Community/Clique')
    plt.legend(['Clique', 'Community'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    output_path = working_dir + 'median_overall_internal_distance_by_community_size'
    plt.savefig(output_path)
    plt.close()

    ax = clq_df.plot(kind='scatter', marker='x', color='g', x='size', y='jen_mean')
    comm_df.plot(kind='scatter', color='b', x='size', y='jen_mean', ax=ax, 
                 ylim=(0, np.log(2) + .001), xlim=(-5, comm_df['size'].max() + 5),
                 title='Average Community/Clique Similarity by Size')
    print('Drawing overall average clique & community divergence by size graph')
    plt.ylabel('Median Jensen Shannon Divergence\n')
    plt.xlabel('Size of Community/Clique')
    plt.legend(['Clique', 'Community'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.subplots_adjust(bottom=0.2)
    output_path = working_dir + 'average_overall_internal_distance_by_community_size'
    plt.savefig(output_path)
    plt.close()

def user_topic_distribution_graph(ovrwrt, community):
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
        if not os.path.exists(output_path + user + '.png') or ovrwrt:
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

def restore_original_dataset(community):
    working_dir = community.strip('/').split('/')[0]
    if os.path.exists(community + '/community_doc_vecs.json.bak'):
        copyfile(community + '/community_doc_vecs.json.bak', community + '/community_doc_vecs.json')
        os.remove(community + '/community_doc_vecs.json.bak')

def delete_inactive_communities(community):
    '''
    if a clique has 1 or less active members then the community and the clique must be
    removed from the dataset
    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if len(comm_doc_vecs) <= 1:
        if os.path.exists(community + '/community_doc_vecs.json'):
            print('Removing: ' + str(community))
            os.remove(community + '/community_doc_vecs.json')
        if os.path.exists(community.replace('clique', 'community') + '/community_doc_vecs.json'):
            print('Removing: ' + str(community.replace('clique', 'community')))
            os.remove(community.replace('clique', 'community') + '/community_doc_vecs.json')

def delete_inactive_users(community):
    '''
    removes users from the dataset if the amount of times they tweeted is less than 10

    '''
    comm_doc_vecs = open_community_document_vectors_file(community + '/community_doc_vecs.json')
    if comm_doc_vecs:
        copyfile(community + '/community_doc_vecs.json', community + '/community_doc_vecs.json.bak') 

    with open('../topic_modeling/inactive_users.json', 'r') as infile:
        inactive_users = json.load(infile)

    for user in inactive_users:
        if user in comm_doc_vecs:
            print("Removing inactive user: " + user)
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
            yield(os.path.join(path, community))
        break

def main():
    parser = argparse.ArgumentParser(description="""Calculate various distances of users in communities and use those distances to plot graphs
                                                    to run all without disturbing the dataset:
                                                      python plot_distances.py -w working_dir/ -iedtms
                                                 """)
    parser.add_argument('-w', '--working_dir', required=True, action='store', dest='working_dir', help='Location of working directory')
    parser.add_argument('-o', action='store_true', help='Omit inactive users and communities from calculations/graphs')
    parser.add_argument('-r', action='store_true', help='Revert to original dataset (use if removed inactive users and want to restore dataset')
    parser.add_argument('-i', action='store_true', help='Draw internal distance graphs for each user')
    parser.add_argument('-e', action='store_true', help='Draw external distance graphs for each user')
    parser.add_argument('-d', action='store_true', help='Draw internal vs external distance graphs for each user')
    parser.add_argument('-t', action='store_true', help='Draw topic distribution graphs for each user')
    parser.add_argument('-m', action='store_true', help='Draw median distance graphs for communities')
    parser.add_argument('-s', action='store_true', help='Draw distance vs size distribution graphs for communities')
    parser.add_argument('-x', '--overwrite', action='store_true', dest='ovrwrt', help='Use this option to overwrite current graphs')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    columns=['cid', 'user_a', 'user_b', 'cos', 'euc', 'hel', 'jen']
    agg_cols = ['cos_med', 'cos_mean', 'euc_med', 'euc_mean', 'hel_med', 'hel_mean', 'jen_med', 'jen_mean']
    agg_metrics = OrderedDict([('cos',['median', 'mean']), ('euc', ['median', 'mean']), ('hel', ['median', 'mean']), ('jen', ['median', 'mean'])])
    total_work = len([dir for dir in os.listdir('info_map_comms/') if os.path.isdir(os.path.join('info_map_comms', dir))])
    if args.o:
        print('Removing inactive users')
        for _ in tqdm.tqdm(pool.imap_unordered(delete_inactive_users, dir_to_iter(args.working_dir)), total=total_work): pass
        print('Removing inactive communities')
        for _ in tqdm.tqdm(pool.imap_unordered(delete_inactive_communities, dir_to_iter(args.working_dir)), total=total_work): pass
    if args.r:
        print('Restoring original dataset')
        for _ in tqdm.tqdm(pool.imap(restore_original_dataset, dir_to_iter(args.working_dir)), total=total_work): pass
    if args.i:
        print('Calculating internal user distances')
        for community in tqdm.tqdm(dir_to_iter(args.working_dir), total=total_work):
            calculate_internal_distances(community)
        print('Compiling dataframe from result')
        conn = MySQLdb.connect(host='localhost', user='root', passwd='yetanotherpassword', db='distances')
        df = pd.read_sql_query('SELECT cid, user_a, user_b, cos, euc, hel, jen FROM internal_distances;', conn)
        conn.close()
        df.to_csv(os.path.join(args.working_dir, 'internal_distances'), sep='\t')
#        func = partial(user_distance_graphs, df, 'internal', args.ovrwrt)
#        pool.map(func, dir_to_iter(args.working_dir))
        df = df.groupby(['cid']).agg(agg_metrics)
        df.columns = (agg_cols)
        df.to_csv(os.path.join(args.working_dir, 'internal_aggregated_distances'), sep='\t', index_label='cid')
        pool.close()
    if args.e:
        print('Calculating external user distances')
        for community in tqdm.tqdm(dir_to_iter(args.working_dir), total=total_work):
            calculate_external_distances(community)
        print('Compiling dataframe from result')
        conn = MySQLdb.connect(host='localhost', user='root', passwd='yetanotherpassword', db='distances')
        df = pd.read_sql_query('SELECT cid, user_a, user_b, cos, euc, hel, jen FROM external_distances;', conn)
        conn.close()
        df.to_csv(os.path.join(args.working_dir, 'internal_distances'), sep='\t')
#        func = partial(user_distance_graphs, df, 'external', args.ovrwrt)
#        pool.map(func, dir_to_iter(args.working_dir))
        df = df.groupby(['cid']).agg(agg_metrics)
        df.columns = (agg_cols)
        df.to_csv(os.path.join(args.working_dir, 'external_aggregated_distances'), sep='\t', index_label='cid')
        pool.close()
    if args.d:
        func = partial(user_internal_external_graphs, args.ovrwrt)
        pool.map(func, dir_to_iter(args.working_dir))
        pool.close()
    if args.t:
        func = partial(user_topic_distribution_graph, args.ovrwrt)
        pool.map(func, dir_to_iter(args.working_dir))
        pool.close()

    pool.terminate()
 
    if args.m:
        community_aggregated_internal_external_distance_graphs(args.working_dir)
    if args.s:
        aggregated_internal_distance_by_community_size_graph(args.working_dir)

if __name__ == '__main__':
    sys.exit(main())
