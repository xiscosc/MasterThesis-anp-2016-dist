from __future__ import print_function

import os
import sys
import pickle
import argparse


parser = argparse.ArgumentParser('Create Adjective and Noun lists from a list of ANPs')
parser.add_argument('--anp_file', help='Text file with the list of ANPs')
parser.add_argument('--output_dir', help='Directory where the output files will be stored')
parser.add_argument('--prefix',
                    help='Prefix for the output files, named prefix_noun_list.txt and prefix_adjective_list.txt')

args = parser.parse_args()


# Open ANP file
with open(args.anp_file, 'r') as f:
    anps = [line.strip() for line in f.readlines()]


# Create lists for Nouns and Adjectives
nouns = []
adjectives = []

for anp in anps:
    a, n = anp.split('_')
    if '%s\n' % a not in adjectives:
        adjectives.append('%s\n' % a)
    if '%s\n' % n not in nouns:
        nouns.append('%s\n' % n)


# Store outputs
with open(os.path.join(args.output_dir, '%s_adjective_list.txt' % args.prefix), 'w') as f:
    f.writelines(adjectives)


with open(os.path.join(args.output_dir, '%s_noun_list.txt' % args.prefix), 'w') as f:
    f.writelines(nouns)
