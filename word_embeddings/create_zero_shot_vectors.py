from __future__ import print_function

import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser('Compare A/N word2vec vectors addition, concatenation and product methods')
parser.add_argument('--ground_truth_dir', help='Directory containing class lists')
parser.add_argument('--vectors_dir', help='Directory containing all pickle files')
parser.add_argument('--flickr', default=False, action='store_true', help="Whether to use Flickr instead of Google News")

args = parser.parse_args()


def load_vectors(vectors_file, ground_truth_file):
    """
    Loads word2vec vectors, normalizes them and creates a matrix where mat[i,:] is the i-th normalized vector
    :param vectors_file: path to the pickle file with the vectors
    :param ground_truth_file: path to the list of classes
    :return: numpy array matrix with the vectors, list of classes
    """
    # Load data from files
    with open(vectors_file, 'rb') as f:
        vectors_dict = pickle.load(f, encoding='latin1')
    with open(ground_truth_file, 'r') as f:
        ground_truth = [line.strip() for line in f.readlines()]

    vectors_dict['artgallery'] = vectors_dict['gallery']

    # Create list of vectors in the same order as the labels and normalize them
    all_vectors = [vectors_dict[label] / np.linalg.norm(vectors_dict[label]) for label in ground_truth]

    # Create matrix with all the vectors
    return np.array(all_vectors), ground_truth


# Generate ANP vectors for the first time
# Load Adj data
if args.flickr:
    suffix = '_flickr'
    adj_vectors, adj_gt = load_vectors(
        os.path.join(args.vectors_dir, 'zero_shot_flickr_w5_d300_c10-cwvecs-adjectives.pickle'),
        os.path.join(args.ground_truth_dir, 'zero_shot_adjective_list.txt')
    )

    # Load Noun data
    noun_vectors, noun_gt = load_vectors(
        os.path.join(args.vectors_dir, 'zero_shot_flickr_w5_d300_c10-cwvecs-nouns.pickle'),
        os.path.join(args.ground_truth_dir, 'zero_shot_noun_list.txt')
    )
else:
    suffix = ''
    adj_vectors, adj_gt = load_vectors(
        os.path.join(args.vectors_dir, 'zero_shot_GoogleNews-vectors-negative300-adjectives.pickle'),
        os.path.join(args.ground_truth_dir, 'zero_shot_adjective_list.txt')
    )

    # Load Noun data
    noun_vectors, noun_gt = load_vectors(
        os.path.join(args.vectors_dir, 'zero_shot_GoogleNews-vectors-negative300-nouns.pickle'),
        os.path.join(args.ground_truth_dir, 'zero_shot_noun_list.txt')
    )

# Load ANP data
with open(os.path.join(args.ground_truth_dir, 'zero_shot_class_list.txt'), 'r') as f:
    anp_gt = [line.strip() for line in f.readlines()]

# Build matrix of ANP vectors
anp_vectors_add = [None] * len(anp_gt)
anp_vectors_concat = [None] * len(anp_gt)
anp_vectors_prod = [None] * len(anp_gt)
for i, anp in enumerate(anp_gt):
    adj, noun = anp.split('_')
    adj_index = adj_gt.index(adj)
    noun_index = noun_gt.index(noun)
    anp_vectors_add[i] = adj_vectors[adj_index, :] + noun_vectors[noun_index, :]
    anp_vectors_concat[i] = np.concatenate([adj_vectors[adj_index, :], noun_vectors[noun_index, :]])
    anp_vectors_prod[i] = np.multiply(adj_vectors[adj_index, :], noun_vectors[noun_index, :])

# Build matrix with numpy
anp_vectors_add = np.array(anp_vectors_add)
anp_vectors_concat = np.array(anp_vectors_concat)
anp_vectors_prod = np.array(anp_vectors_prod)

# Check dimensions
print('Dimensions after adding: ', np.shape(anp_vectors_add))
print('Dimensions after concatenating: ', np.shape(anp_vectors_concat))
print('Dimensions after multiplying: ', np.shape(anp_vectors_prod))


with open(os.path.join(args.vectors_dir, 'anp_vectors_add%s.pickle' % suffix), 'rb') as f:
    anp_vectors_add_original = pickle.load(f, encoding='latin1')

with open(os.path.join(args.vectors_dir, 'anp_vectors_concat%s.pickle' % suffix), 'rb') as f:
    anp_vectors_concat_original = pickle.load(f, encoding='latin1')

with open(os.path.join(args.vectors_dir, 'anp_vectors_prod%s.pickle' % suffix), 'rb') as f:
    anp_vectors_prod_original = pickle.load(f, encoding='latin1')


# Store matrices
with open(os.path.join(args.vectors_dir, 'zero_shot_anp_vectors_add%s.pickle' % suffix), 'wb') as f:
    pickle.dump(np.concatenate((anp_vectors_add_original, anp_vectors_add)), f)

with open(os.path.join(args.vectors_dir, 'zero_shot_anp_vectors_concat%s.pickle' % suffix), 'wb') as f:
    pickle.dump(np.concatenate((anp_vectors_concat_original, anp_vectors_concat)), f)

with open(os.path.join(args.vectors_dir, 'zero_shot_anp_vectors_prod%s.pickle' % suffix), 'wb') as f:
    pickle.dump(np.concatenate((anp_vectors_prod_original, anp_vectors_prod)), f)
