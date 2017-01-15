from __future__ import print_function

import os
import sys
import argparse
from PIL import Image
from multiprocessing import Process


def list_emotion(source_dir, dest_dir):
    """
    List images to convert
    :param source_dir:
    :param dest_dir:
    :return: list with (source_image, dest_image) tuples
    """
    source_images = os.listdir(source_dir)
    return [(os.path.join(source_dir, image_file), os.path.join(dest_dir, image_file)) for image_file in source_images]


def process_pairs(thread_id, pairs):
    """
    Loads and converts a list of (source, dest) pairs
    :param thread_id:
    :param pairs:
    :return:
    """
    for index, (source, dest) in enumerate(pairs):
        try:
            img = Image.open(source)
            img.convert('RGB').save(dest, 'jpeg')
        except:
            print('Failed to convert %s' % source)
            sys.stdout.flush()
        if index % 2000 == 0:
            print('Thread %d: processed %d/%d images' % (thread_id, index, len(pairs)))
            sys.stdout.flush()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(args):
    # List all emotions
    emotions = [f for f in os.listdir(args.source_dir) if 'ground_truth' not in f]

    # List all images
    print('Listing images...')
    sys.stdout.flush()
    all_pairs = []
    for emotion in emotions:
        all_pairs.extend(list_emotion(os.path.join(args.source_dir, emotion), os.path.join(args.dest_dir, emotion)))

    # Split among threads
    print('Found %d images to convert' % len(all_pairs))
    sys.stdout.flush()
    chunk_size = int(len(all_pairs) / args.n_threads)
    chunks_generator = chunks(all_pairs, chunk_size)

    # Launch threads to download the files
    print('Starting conversion with %d threads' % args.n_threads)
    sys.stdout.flush()
    processes = [Process(target=process_pairs, args=(i, chunks_generator.next())) for i in range(args.n_threads)]
    for p in processes:
        p.start()

    # Wait until all the threads finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert downloaded images to a proper JPEG format')
    parser.add_argument('--source_dir', help='Source directory')
    parser.add_argument('--dest_dir', help='Destination directory')
    parser.add_argument('--n_threads', type=int, default=1, help='Number of threads downloading images in parallel')
    arguments = parser.parse_args()

    main(arguments)
