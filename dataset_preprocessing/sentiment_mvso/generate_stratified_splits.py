from __future__ import print_function

import os
import sys
import random
import argparse


def split_list(input_list, split_fraction):
    """
    Splits a list into two
    :param input_list:
    :param split_fraction:
    :return:
    """
    split_index = int(split_fraction * len(input_list))
    return input_list[:split_index], input_list[split_index:]


def main(args):
    all_images_per_class = {}
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            filename, label = line.strip().split()
            label = int(label)
            if label not in all_images_per_class.keys():
                all_images_per_class[label] = [filename]
            else:
                all_images_per_class[label].append(filename)

    train_images_per_class = {}
    val_images_per_class = {}

    for label in all_images_per_class.keys():
        train_images_per_class[label], val_images_per_class[label] = split_list(all_images_per_class[label],
                                                                                args.train_split)

    train_lines = []
    val_lines = []
    for label in all_images_per_class.keys():
        train_lines.extend(['%s %d\n' % (filename, label) for filename in train_images_per_class[label]])
        val_lines.extend(['%s %d\n' % (filename, label) for filename in val_images_per_class[label]])

    with open(os.path.join(args.output_dir, 'mvso_sentiment_train.txt'), 'w') as f:
        f.writelines(train_lines)

    with open(os.path.join(args.output_dir, 'mvso_sentiment_val.txt'), 'w') as f:
        f.writelines(val_lines)

    print('Train images: %d' % len(train_lines))
    print('Validation images: %d' % len(val_lines))
    print('\nImages per class:')
    for label in all_images_per_class.keys():
        print('\t%d: %d' % (label, len(all_images_per_class[label])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates random stratified train/val splits for MVSO Sentiment')
    parser.add_argument('--train_split', dest='train_split', type=float,
                        help='Fraction of the dataset used for training')
    parser.add_argument('--input_file', dest='input_file',
                        help='File with (file, label) tuples')
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory where the generated train.txt and val.txt files will be stored')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
                        help='Shuffle the path lists in the output files' +
                             ' (recommended if TFRecord files need to be generated)')

    arguments = parser.parse_args()
    main(arguments)
