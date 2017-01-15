from __future__ import absolute_import, print_function

import os
import sys
import pickle
import argparse
import numpy as np
import gensim.models

sys.path.insert(0, '..')
from helpers.monitoring import Verbose

parser = argparse.ArgumentParser('Create label->vector dictionary for the ground truth')
parser.add_argument('--ground_truth', help='Text file with the list of labels')
parser.add_argument('--vectors', help='word2vec bin file')
parser.add_argument('--output_dir', help='Directory where the dictionary will be saved')

args = parser.parse_args()


# Some words are spelled different in US/UK English. Besides, there are proper names that do not appear in word2vec.
exceptions = {
    'wellcome': 'welcome',
    'coloured': 'colored',
    'harbour': 'harbor',
    'theatre': 'theater',
    'langdale': 'valley',
    'colouring': 'coloring',
    'archaeology': 'archeology',
    'centre': 'center',
    'exsposure': 'exposure',
    'judgement': 'judgment',
    'labour': 'labor',
    'civilisation': 'civilization',
    'colourful': 'colorful',
    'armoured': 'armored',
    'unrecognisable': 'unrecognizable'
}


with Verbose('Loading word2vec model...'):
    model = gensim.models.Word2Vec.load_word2vec_format(args.vectors, binary=True)

with open(args.ground_truth, 'r') as f, Verbose('Loading ground truth...'):
    ground_truth_labels = [line.strip() for line in f.readlines()]

all_vectors_dict = {}
for label in ground_truth_labels:
    try:
        if label in exceptions: word = exceptions[label]
        else: word = label
        all_vectors_dict[label] = model[word]
    except KeyError:
        print('Word not found: %s' % word)

with Verbose('Saving file...'):
    output_file = '%s.pickle' % args.vectors.split('/')[-1].rsplit('.', 1)[0]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, output_file), 'wb') as f:
        pickle.dump(all_vectors_dict, f)

print('Dictionary saved to %s' % os.path.join(args.output_dir, output_file))
