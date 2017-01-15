from __future__ import print_function

import os
import argparse
from multiprocessing import Process


def process_emotion(csv_files, images_root_dir, decision_fn, output_dir):
    """
    Processes all the CSV for an emotion
    :param csv_files: list with the CSV files for the emotion being processed
    :param images_root_dir: directory where the images have been downloaded
    :param decision_fn: function that returns a boolean given the number of positive and negative answers from AMT
    :param output_dir: directory where the resulting list will be stored
    :return:
    """
    # Read all files
    lines = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            lines.extend(f.readlines())

    # Output list
    lines_to_write = []

    # Process all images
    for line in lines:
        emotion, url, negative, positive = line.split(',')
        filename = url.split('/')[-1]
        positive = int(positive)
        negative = int(negative)

        if os.path.exists(os.path.join(images_root_dir, emotion, filename)) and decision_fn(positive, negative):
            line = '%s\n' % filename
            if line not in lines_to_write:
                lines_to_write.append(line)

    # Write to output file
    with open(os.path.join(output_dir, '%s.txt' % emotion), 'w') as f:
        f.writelines(lines_to_write)

    print('%s: %d/%d images have been kept' % (emotion, len(lines_to_write), len(lines)))


def all_agree(positive, negative):
    return (positive > 0) and (negative == 0)


def majority_voting(positive, negative):
    return positive > negative


def main():
    parser = argparse.ArgumentParser('Generates ground truth from CSV')
    parser.add_argument('--csv_root_dir', help='Directory containing all CSV files')
    parser.add_argument('--images_root_dir', help='Directory containing the downloaded images: root_dir/emotion/image')
    parser.add_argument('--output_dir', help='Directory where the lists will be stored: output_root_dir/emotion.txt')
    parser.add_argument('--all_agree', action='store_true', default=False,
                        help='Whether to only select images that built consensus among all the annotators')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    emotions = [directory for directory in os.listdir(args.images_root_dir) if directory != 'ground_truth']
    all_csv_files = [os.path.join(args.csv_root_dir, f) for f in os.listdir(args.csv_root_dir) if f.endswith('csv')]

    csv_files_per_emotion = [filter(lambda x: x.__contains__(emotion), all_csv_files) for emotion in emotions]

    if args.all_agree:
        decision_fn = all_agree
    else:
        decision_fn = majority_voting

    processes = [Process(target=process_emotion, args=(csv_files, args.images_root_dir, decision_fn, args.output_dir))
                 for csv_files in csv_files_per_emotion]

    # Launch all processes
    for p in processes:
        p.start()

    # Wait until all the threads finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
