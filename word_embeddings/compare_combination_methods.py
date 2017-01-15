from __future__ import print_function

import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser('Compare A/N word2vec vectors addition, concatenation and product methods')
parser.add_argument('--create_anp_pickles', action='store_true', default=False,
                    help="Whether to generate or load ANP vectors")
parser.add_argument('--vectors_dir', help='Directory containing all pickle files')
parser.add_argument('--ground_truth_dir', help='Directory containing class lists')
parser.add_argument('--anps', nargs='+', help='Which ANPs to check', required=True)
parser.add_argument('--n_neighbors', type=int, default=5, help='How many neighbors to list')

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
        vectors_dict = pickle.load(f)  # , encoding='latin1')
    with open(ground_truth_file, 'r') as f:
        ground_truth = [line.strip() for line in f.readlines()]

    # Create list of vectors in the same order as the labels and normalize them
    all_vectors = [vectors_dict[label]/np.linalg.norm(vectors_dict[label]) for label in ground_truth]

    # Create matrix with all the vectors
    return np.array(all_vectors), ground_truth


# Generate ANP vectors for the first time
if args.create_anp_pickles:
    # Load Adj data
    adj_vectors, adj_gt = load_vectors(
        os.path.join(args.vectors_dir, 'GoogleNews-vectors-negative300-adjectives.pickle'),
        os.path.join(args.ground_truth_dir, 'adj_list.txt')
    )

    # Load Noun data
    noun_vectors, noun_gt = load_vectors(
        os.path.join(args.vectors_dir, 'GoogleNews-vectors-negative300-nouns.pickle'),
        os.path.join(args.ground_truth_dir, 'noun_list.txt')
    )

    # Load ANP data
    with open(os.path.join(args.ground_truth_dir, 'anp_list.txt'), 'r') as f:
        anp_gt = [line.strip() for line in f.readlines()]

    # Build matrix of ANP vectors
    anp_vectors_add = [None] * len(anp_gt)
    anp_vectors_concat = [None] * len(anp_gt)
    anp_vectors_prod = [None] * len(anp_gt)
    anp_vectors_max = [None] * len(anp_gt)
    for i, anp in enumerate(anp_gt):
        adj, noun = anp.split('_')
        adj_index = adj_gt.index(adj)
        noun_index = noun_gt.index(noun)
        anp_vectors_add[i] = adj_vectors[adj_index, :] + noun_vectors[noun_index, :]
        anp_vectors_concat[i] = np.concatenate([adj_vectors[adj_index, :], noun_vectors[noun_index, :]])
        anp_vectors_prod[i] = np.multiply(adj_vectors[adj_index, :], noun_vectors[noun_index, :])
        anp_vectors_max[i] = np.maximum(adj_vectors[adj_index, :], noun_vectors[noun_index, :])

    # Build matrix with numpy
    anp_vectors_add = np.array(anp_vectors_add)
    anp_vectors_concat = np.array(anp_vectors_concat)
    anp_vectors_prod = np.array(anp_vectors_prod)
    anp_vectors_max = np.array(anp_vectors_max)

    # Check dimensions
    print('Dimensions after adding: ', np.shape(anp_vectors_add))
    print('Dimensions after concatenating: ', np.shape(anp_vectors_concat))
    print('Dimensions after multiplying: ', np.shape(anp_vectors_prod))
    print('Dimensions after max: ', np.shape(anp_vectors_max))

    # Store matrices
    with open(os.path.join(args.vectors_dir, 'anp_vectors_add.pickle'), 'wb') as f:
        pickle.dump(anp_vectors_add, f)

    with open(os.path.join(args.vectors_dir, 'anp_vectors_concat.pickle'), 'wb') as f:
        pickle.dump(anp_vectors_concat, f)
        
    with open(os.path.join(args.vectors_dir, 'anp_vectors_prod.pickle'), 'wb') as f:
        pickle.dump(anp_vectors_prod, f)

    with open(os.path.join(args.vectors_dir, 'anp_vectors_max.pickle'), 'wb') as f:
        pickle.dump(anp_vectors_max, f)

# Load pre-computed ANP vectors
else:
    with open(os.path.join(args.ground_truth_dir, 'anp_list.txt'), 'r') as f:
        anp_gt = [line.strip() for line in f.readlines()]

    with open(os.path.join(args.vectors_dir, 'anp_vectors_add.pickle'), 'rb') as f:
        anp_vectors_add = pickle.load(f)  # , encoding='latin1')

    with open(os.path.join(args.vectors_dir, 'anp_vectors_concat.pickle'), 'rb') as f:
        anp_vectors_concat = pickle.load(f)  # , encoding='latin1')

    with open(os.path.join(args.vectors_dir, 'anp_vectors_prod.pickle'), 'rb') as f:
        anp_vectors_prod = pickle.load(f)  # , encoding='latin1')

    with open(os.path.join(args.vectors_dir, 'anp_vectors_max.pickle'), 'rb') as f:
        anp_vectors_max = pickle.load(f)  # , encoding='latin1')


# Find NN for each ANP
print('\n\n\n')
for anp in args.anps:
    anp_ind = anp_gt.index(anp)

    # Compute NN for the addition method
    sample_add = anp_vectors_add[anp_ind, :]
    similarity_add = anp_vectors_add.dot(sample_add)
    nn_add_indices = similarity_add.argsort()[::-1][1:args.n_neighbors+1]
    str_add = '\tAddition - most similar: '
    for ind in nn_add_indices: str_add += ' %s,' % anp_gt[ind]
    str_add = str_add[:-1]
    nn_add_indices = similarity_add.argsort()[0:args.n_neighbors]
    str_add += '\n\tAddition - least similar:'
    for ind in nn_add_indices: str_add += ' %s,' % anp_gt[ind]
    str_add = str_add[:-1]

    # Compute NN for the concatenation method
    sample_concat = anp_vectors_concat[anp_ind, :]
    similarity_concat = anp_vectors_concat.dot(sample_concat)
    nn_concat_indices = similarity_concat.argsort()[::-1][1:args.n_neighbors + 1]
    str_concat = '\tConcat - most similar:   '
    for ind in nn_concat_indices: str_concat += ' %s,' % anp_gt[ind]
    str_concat = str_concat[:-1]
    nn_concat_indices = similarity_concat.argsort()[0:args.n_neighbors]
    str_concat += '\n\tConcat - least similar:  '
    for ind in nn_concat_indices: str_concat += ' %s,' % anp_gt[ind]
    str_concat = str_concat[:-1]

    # Compute NN for the product method
    sample_prod = anp_vectors_prod[anp_ind, :]
    similarity_prod = anp_vectors_prod.dot(sample_prod)
    nn_prod_indices = similarity_prod.argsort()[::-1][1:args.n_neighbors + 1]
    str_prod = '\tProduct - most similar:  '
    for ind in nn_prod_indices: str_prod += ' %s,' % anp_gt[ind]
    str_prod = str_prod[:-1]
    nn_prod_indices = similarity_prod.argsort()[0:args.n_neighbors]
    str_prod += '\n\tProduct - least similar: '
    for ind in nn_prod_indices: str_prod += ' %s,' % anp_gt[ind]
    str_prod = str_prod[:-1]

    # Compute NN for the max method
    sample_max = anp_vectors_max[anp_ind, :]
    similarity_max = anp_vectors_max.dot(sample_max)
    nn_max_indices = similarity_max.argsort()[::-1][1:args.n_neighbors + 1]
    str_max = '\tMaximum - most similar:  '
    for ind in nn_max_indices: str_max += ' %s,' % anp_gt[ind]
    str_max = str_max[:-1]
    nn_max_indices = similarity_max.argsort()[0:args.n_neighbors]
    str_max += '\n\tMaximum - least similar: '
    for ind in nn_max_indices: str_max += ' %s,' % anp_gt[ind]
    str_max = str_max[:-1]

    print('Top similarities for %s:\n%s\n%s\n%s\n%s\n' % (anp, str_add, str_concat, str_prod, str_max))

print('\n\n\n')
