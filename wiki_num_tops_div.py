import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import csv

def draw_graph()
    df = pd.read_csv('wiki_num_topics_divergence', sep='\t', header=None, index_col=0, names=['num_topics', 'clq_int', 'clq_ext', 'com_int', 'com_ext'])
    ax = df.plot(kind='line', marker='x', color='g', x=df.index, y='com_int')
    df.plot(kind='line', marker='o', color='b', x=df.index, y='com_ext', ax=ax, ylim=(0, np.log(2) + .001), xlim=(24, df.index.max() + 1), 
            title='Overall Mean Divergence Per Number of Topics\nUsing CAA and Wiki Model')
    plt.ylabel('Jensen Shannon Divergence')
    plt.xlabel('Number of Topics')
    plt.legend(['Internal', 'External'], loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.xticks([25, 50, 75, 100])
    plt.subplots_adjust(bottom=0.2)
    plt.grid()
    plt.savefig('wiki_num_topics_divergence.eps', format='eps')
    plt.savefig('wiki_num_topics_divergence')
    plt.close()

def overall_average_divergence_per_model(user_topics_dir, lda_loc):
    print('Writing overall internal & external average community distance for each topic model')
    lda = models.LdaModel.load(lda_loc)
    clique_int_dists = []
    clique_ext_dists = []
    comm_int_dists = []
    comm_ext_dists = []
    fieldnames = ['metric', 'distance']
    for path, dirs, files in os.walk(user_topics_dir):
        for community in dirs:
            with open(path + community + '/distance_info/community_average_distances', 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                if 'clique' in community:
                    clique_int_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
                else:
                    comm_int_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]

            with open(path + community + '/distance_info/external_average_distances', 'r') as infile:
                csv_reader = csv.DictReader(infile, delimiter='\t', fieldnames=fieldnames)
                if 'clique' in community:
                    clique_ext_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
                else:
                    comm_ext_dists += [float(row['distance']) for row in csv_reader if row['metric'] == 'jensen_shannon' and row['distance']]
        break

    if 'twitter' not in user_topics_dir:
        with open('wiki_num_topics_divergence', 'a') as outfile:
            outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(lda.num_topics, np.average(clique_int_dists), np.average(clique_ext_dists), np.average(comm_int_dists), np.average(comm_ext_dists)))

def main():
    lda_dir = os.path.join('data', 'wiki')
    param_dirs = [('user_topics_wiki_25/', os.path.join(lda_dir, 'lda_25_lem_5_pass.model')), 
                  ('user_topics_wiki_50/', os.path.join(lda_dir, 'lda_50_lem_5_pass.model')),
                  ('user_topics_wiki_75/', os.path.join(lda_dir, 'lda_75_lem_5_pass.model')), 
                  ('user_topics_wiki_100/', os.path.join(lda_dir, 'lda_100_lem_5_pass.model'))]
    for user_topics_dir, lda_loc in param_dirs:
        overall_average_divergence_per_model(user_topics_dir, lda_loc)

if __name__ == '__main__':
    sys.exit(main())


