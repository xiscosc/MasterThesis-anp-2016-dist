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
    # List all files
    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.txt')]

    all_classes = []
    all_train_images = []
    all_val_images = []

    for class_id, text_file in enumerate(all_files):
        emotion = text_file.split('/')[-1].split('.')[0]
        with open(text_file, 'r') as f:
            lines = ['%s %d\n' % (line.strip(), class_id) for line in f.readlines()]
        random.shuffle(lines)
        train, val = split_list(lines, args.train_split)
        all_train_images.extend(train)
        all_val_images.extend(val)
        all_classes.append('%s\n' % emotion)

    if args.shuffle:
        random.shuffle(all_train_images)
        random.shuffle(all_val_images)

    # Write outputs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
        f.writelines(all_train_images)

    with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
        f.writelines(all_val_images)

    with open(os.path.join(args.output_dir, 'class_list.txt'), 'w') as f:
        f.writelines(all_classes)

    print('Saved files to %s' % args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates random stratified train/val splits for Rochester Emotion')
    parser.add_argument('--train_split', dest='train_split', type=float,
                        help='Fraction of the dataset used for training')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='Directory with emotion.txt files')
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory where the generated train.txt, val.txt and class_list.txt files will be stored')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
                        help='Shuffle the path lists in the output files' +
                             ' (recommended if TFRecord files need to be generated)')

    arguments = parser.parse_args()
    main(arguments)
