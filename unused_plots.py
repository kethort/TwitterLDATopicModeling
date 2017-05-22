def num_users_distance_range_graph(community):
    '''
    bar graph showing occurrences where users in community are distant from other users in the same community 

    '''
    comm_doc_vecs = open_community_document_vectors_file(community)

    if(len(comm_doc_vecs) <= 1):
        return

    jsd_path = community + '/num_users_distance_range_graphs/jensen_shannon/'

    if not os.path.exists(os.path.dirname(jsd_path)):
        os.makedirs(os.path.dirname(jsd_path), 0o755)

    internal_data = community + '/user_to_internal_users_graphs/jensen_shannon/'
    external_data = community + '/user_to_external_users_graphs/jensen_shannon/'

    distances = []
    fieldnames = ['user_1', 'user_2', 'distance']
    print('Drawing number users as grouped Jensen Shannon' + ' distance graph for: ' + str(community))
    for user in comm_doc_vecs:
        if not os.path.exists(output_dir + user + '.png'):
            with open(internal_data + user, 'r') as plot_file:
                csv_reader = csv.DictReader(plot_file, delimiter='\t', fieldnames=fieldnames)
                distances = [float(row['distance']) for row in csv_reader]
            num_internal_users = bin_jsd_by_range(distances)
            with open(external_data + user, 'r') as plot_file:
                csv_reader = csv.DictReader(plot_file, delimiter='\t', fieldnames=fieldnames)
                distances = [float(row['distance']) for row in csv_reader]
            num_external_users = bin_jsd_by_range(distances)
            plt.xlabel('Jensen Shannon Divergence')
            plt.title('Internal/External Divergence Range from User: ' + str(user))
            objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')

            width = 0.4
            x_axis = np.arange(len(objects))
            plt.bar(x_axis - 0.2, num_internal_users, width, alpha=0.4, color='r')
            plt.bar(x_axis + 0.2, num_external_users, width, alpha=0.4, color='b')
            plt.ylabel('Number of users')
            plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
            plt.xticks(x_axis, objects, rotation=45, fontsize='small')
            plt.subplots_adjust(bottom=0.2)
            plt.xlim([-1, len(objects)])
            plt.ylim([0, len(comm_doc_vecs)])
            plt.savefig(output_dir + user)
            plt.close()

def median_similarity_clique_community_size_graph(working_dir):
    '''
    graph displaying the overall median JSD compared to community size by binning

    '''
    clq_x_axis = []
    comm_x_axis = []
    int_clq_dists = []
    ext_clq_dists = []
    int_comm_dists = []
    ext_comm_dists = []
    clq_divs = defaultdict(list)
    comm_divs = defaultdict(list)

    fieldnames = ['metric', 'distance']
    print('Drawing clique/community size to median distance graph using Jensen Shannon divergence')

    for path, dirs, files in os.walk(working_dir):
        for community in sorted(dirs):
            with open(path + community + '/community_doc_vecs.json', 'r') as comm_doc_vecs_file:
                comm_doc_vecs = json.load(comm_doc_vecs_file)

            with open(path + community + '/distance_info/median_distance_of_each_community', 'r') as avg_dist_file:
                csv_reader = csv.DictReader(avg_dist_file, delimiter='\t', fieldnames=fieldnames)
                for row in csv_reader:
                    if(row['metric'] == 'jensen_shannon'):
                        if 'clique' in community:
                            clq_divs[len(comm_doc_vecs)].append(float(row['distance']))
                            clq_x_axis.append(len(comm_doc_vecs))
                            int_clq_dists.append(float(row['distance']))
                        else:
                            comm_divs[len(comm_doc_vecs)].append(float(row['distance']))
                            comm_x_axis.append(len(comm_doc_vecs))
                            int_comm_dists.append(float(row['distance']))

            with open(path + community + '/distance_info/external_median_distances', 'r') as ext_dist_file:
                csv_reader = csv.DictReader(ext_dist_file, delimiter='\t', fieldnames=fieldnames)
                for row in csv_reader:
                    if(row['metric'] == 'jensen_shannon'):
                        if 'clique' in community:
                            ext_clq_dists = float(row['distance'])
                        else:
                            ext_comm_dists = float(row['distance'])

        binned_jsd_by_range_for_all_graph(working_dir, int_clq_dists, ext_clq_dists, int_comm_dists, ext_comm_dists)
        binned_jsd_by_range_for_two_graph(working_dir, int_clq_dists, int_comm_dists, 'Clique & Community Internal Divergence Distribution', 'Clique', 'Community', 'clq_comm_int_dist_avg_range')
        binned_jsd_by_range_for_two_graph(working_dir, ext_clq_dists, ext_comm_dists, 'Clique & Community External Divergence Distribution', 'Clique', 'Community', 'clq_comm_ext_dist_avg_range')
        binned_jsd_by_range_for_two_graph(working_dir, int_clq_dists, ext_clq_dists, 'Clique Internal & External Divergence Distribution', 'Internal', 'External', 'clq_int_ext_dist_avg_range')
        binned_jsd_by_range_for_two_graph(working_dir, int_comm_dists, ext_comm_dists, 'Community Internal & External Divergence Distribution', 'Internal', 'External', 'comm_int_ext_dist_avg_range')
        overall_median_community_jsd_vs_size_graph(working_dir, clq_divs, comm_divs)

        plt.ylabel('Median Jensen Shannon divergence')
        plt.xlabel('Size of Community')
        plt.title('Median Community Similarity')
        output_path = working_dir + 'community_size_median_divergence'
        plt.scatter(comm_x_axis, int_comm_dists)
        plt.ylim([0, np.log(2) + .001])
        plt.xlim([0, max(comm_x_axis) + 1])
        plt.savefig(output_path)
        plt.close()

        plt.ylabel('Median Jensen Shannon Divergence')
        plt.xlabel('Size of Clique')
        plt.title('Median Clique Divergence')
        output_path = working_dir + 'clique_size_median_divergence'
        plt.scatter(clq_x_axis, int_clq_dists)
        plt.ylim([0, np.log(2) + .001])
        plt.xlim([0, max(clq_x_axis) + 1])
        plt.savefig(output_path)
        plt.close()

        break # restrict depth of folder traversal to 1

def binned_jsd_by_range_for_all_graph(working_dir, int_clq, ext_clq, int_comm, ext_comm):
    '''
    binned frequency of occurences of jensen shannon divergence in 0.1 incremental ranges 
    comparing internal cliques and communities as well as external cliques and communities

    '''
    width = 0.2
    fig, ax = plt.subplots()
    objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')
    x_axis = np.arange(len(objects))
    rects_1 = ax.bar(x_axis, bin_jsd_by_range(int_clq), width, color='r', align='center')
    rects_2 = ax.bar(x_axis + width, bin_jsd_by_range(ext_clq), width, color='g', align='center')
    rects_3 = ax.bar(x_axis + width * 2, bin_jsd_by_range(int_comm), width, color='b', align='center')
    rects_4 = ax.bar(x_axis + width * 4, bin_jsd_by_range(ext_comm), width, color='y', align='center')
    ax.set_xlabel('Jensen Shannon Divergence Distribution')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Clique/Community Divergence Distribution')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(objects, ha='center')
    box = ax.get_position()
    ax.legend((rects_1[0], rects_2[0], rects_3[0], rects_4[0]), ('Clq Int', 'Cliq Ext', 'Comm Int', 'Comm Ext'), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(working_dir + 'clq_comm_distributed_avg_range')
    plt.close(fig)

def bin_jsd_by_range(distances):
    '''
    returns a list of occurences where the overall median community divergences fall
    in the range of 0 to ln(2)

    '''
    bins = np.arange(0, np.log(2) + 0.1, 0.1)
    binned = np.digitize(distances, bins)
    binned = [x - 1 for x in binned]
    return np.bincount(binned)

def binned_jsd_by_range_for_two_graph(working_dir, dists_1, dists_2, title, lg_lbl_1, lg_lbl_2, out_name):
    '''
    binned frequency of occurences of community median jensen shannon divergence in 0.1 incremental ranges 
    comparing two cases, either:

        internal cliques vs community divergences, internal clique vs external clique divergences,
        internal community vs external community divergences or external clique vs external community divergences

    '''
    width = 0.3
    fig, ax = plt.subplots()
    objects = ('[0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '> 0.6')
    x_axis = np.arange(len(objects))
    rects_1 = ax.bar(x_axis, bin_jsd_by_range(dists_1), width, color='r', align='center')
    rects_2 = ax.bar(x_axis + width, bin_jsd_by_range(dists_2), width, color='b', align='center')
    ax.set_xlabel('Jensen Shannon Divergence Distribution')
    ax.set_ylabel('Number of Communities')
    ax.set_title(title)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(objects, ha='center')
    box = ax.get_position()
    ax.legend((rects_1[0], rects_2[0]), (lg_lbl_1, lg_lbl_2), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tick_params(labelsize=10)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(working_dir + out_name)
    plt.close(fig)
