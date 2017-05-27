import numpy as np
import pandas as pd
import sys
import ast
import json
import matplotlib.pyplot as plt

def community_size_distribution():
    df = pd.read_csv('cliques', sep='\] ', engine='python',  header=None)
    sizes = [len(ast.literal_eval(row[0])) for idx, row in df.iterrows()]
    df = pd.read_csv('communities', sep='\] ', engine='python', header=None)
    sizes += [len(ast.literal_eval(row[0])) for idx, row in df.iterrows()]

    x_axis = np.arange(0, max(sizes), 10)
    plt.bar(x_axis, bin_by_x_axis(sizes, x_axis), align='center', width=10, color='y')
    plt.xlabel('Community Size')
    plt.ylabel('Number of Communities')
    plt.title('Community Size Distribution')
    plt.xticks(x_axis, generate_x_ticks(x_axis), rotation='60', ha='right', fontsize='small')
    plt.xlim([-10, np.max(x_axis) + 10])
    plt.tight_layout()
    plt.savefig('community_size_distribution')
    plt.close()

def user_tweet_distribution():
    with open('dnld_tweets/active_users.json', 'r') as infile:
        d = json.load(infile)
    num_tweets = [d[x] for x in d] 
    x_axis = np.arange(0, 3300, 100)
    plt.bar(x_axis, bin_by_x_axis(num_tweets, x_axis), width=100, color='r', align='center')
    plt.xlabel('Number of Tweets')
    plt.xticks(x_axis, generate_x_ticks(x_axis), rotation=60, ha='right', fontsize=8)
    plt.xlim([-100, 3300])
    plt.ylabel('Number of Users')
    plt.title('Tweets per User')
    plt.tight_layout()
    plt.savefig('tweet_distribution')
    plt.close()

def generate_x_ticks(x_axis):
    return [('> ' + str(x_axis[i])) if i == len(x_axis) -1 else (str(x_axis[i]) + ' - ' + str(x_axis[i + 1])) for i in range(0, len(x_axis))]

def bin_by_x_axis(sizes, x_axis):
    return np.bincount([x - 1 for x in np.digitize(sizes, x_axis)])

def main():
    community_size_distribution()
    user_tweet_distribution()

if __name__ == '__main__':
    sys.exit(main())
