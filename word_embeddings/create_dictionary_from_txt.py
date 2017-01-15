from __future__ import absolute_import, print_function

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, '..')
from helpers.monitoring import Verbose

parser = argparse.ArgumentParser('Create label->vector dictionary for the ground truth')
parser.add_argument('--ground_truth', help='Text file with the list of labels')
parser.add_argument('--vectors', help='Text file containing (word, vector) pairs in each line')
parser.add_argument('--output_dir', help='Directory where the dictionary will be saved')

args = parser.parse_args()

with Verbose('Loading ground truth...'):
    with open(args.ground_truth, 'r') as f:
        ground_truth_labels = [line.strip() for line in f.readlines()]

with Verbose('Loading vectors...'):
    with open(args.vectors, 'r') as f:
        lines = f.readlines()[1:]
        all_vectors_dict = {}
        for line in lines:
            word, values = line.strip().split(' ', 1)
            vector = np.array([float(val) for val in values.split()])
            all_vectors_dict[word] = vector

with Verbose('Checking that all classes were found...'):
    for label in ground_truth_labels:
        assert label in all_vectors_dict


with Verbose('Saving file...'):
    output_file = '%s.pickle' % args.vectors.split('/')[-1].rsplit('.', 1)[0]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, output_file), 'wb') as f:
        pickle.dump(all_vectors_dict, f, protocol=0)
