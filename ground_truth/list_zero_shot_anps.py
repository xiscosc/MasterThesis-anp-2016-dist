from __future__ import print_function

import os
import sys
import random
import argparse

from multiprocessing import Process, Lock, Manager


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def select_random_samples(thread_id, chunk, args, all_images, lock):
    local_images = []
    for i, (class_id, anp) in enumerate(chunk):
        anp_folder = os.path.join(args.mvso_images_dir, anp)
        anp_image_set = [os.path.join(anp_folder, img) for img in os.listdir(anp_folder)]
        random.shuffle(anp_image_set)
        local_images.extend(['%s %d\n' % (img, class_id) for img in anp_image_set[:args.num_samples]])
        if i % 50 == 0:
            print('\t[Thread %d] Processed %d ANPs' % (thread_id, i + 1))
    with lock:
        all_images.extend(local_images)
    print('\t[Thread %d] Done!\tProcessed %d ANPs' % (thread_id, i+1))


def list_chunk(thread_id, chunk, args, lock, unseen_anps, samples_per_unseen_anp):
    local_unseen_anps = []
    local_samples_per_unseen_anp = {}
    for i, anp in enumerate(chunk):
        samples = os.listdir(os.path.join(args.mvso_images_dir, anp))
        if len(samples) >= args.num_samples:
            local_samples_per_unseen_anp[anp] = samples
            local_unseen_anps.append(anp)
        if i % 10 == 0:
            print('\t[Thread %d] Processed %d/%d ANPs' % (thread_id, i+1, len(chunk)))
            sys.stdout.flush()
    with lock:
        unseen_anps.extend(local_unseen_anps)
        samples_per_unseen_anp.update(local_samples_per_unseen_anp)
    print('\t[Thread %d] Done! Processed %d/%d ANPs' % (thread_id, i+1, len(chunk)))
    sys.stdout.flush()


def main(args):
    # List unseen ANPs
    with open(args.used_anps, 'r') as f:
        seen_anps = [line.strip() for line in f.readlines()]
    with open(args.all_anps, 'r') as f:
        all_unseen_anps = [line.strip() for line in f.readlines() if line.strip() not in seen_anps]

    # Filter ANPs that do not have enough samples. TODO: avoid calling os.listdir() more than once per ANP
    print('Listing all images per ANP')
    sys.stdout.flush()

    chunks_generator = chunks(all_unseen_anps, int(len(all_unseen_anps) / args.n_threads))
    lock = Lock()
    manager = Manager()
    unseen_anps = manager.list()  # class names for the unseen ANPs
    samples_per_unseen_anp = manager.dict()  # keys: ANPs

    # Launch threads to list all the files
    print('Launching processes (1)')
    sys.stdout.flush()
    processes = [Process(target=list_chunk,
                         args=(i, chunks_generator.next(), args, lock, unseen_anps, samples_per_unseen_anp))
                 for i in range(args.n_threads)]
    for p in processes:
        p.start()

    # Wait until all the threads finish
    for p in processes:
        p.join()

    # Select num_classes ANPs randomly.
    assert len(unseen_anps) >= args.num_classes, 'There are only %d ANPs fulfilling the conditions' % len(unseen_anps)
    random.shuffle(unseen_anps)
    unseen_anps = unseen_anps[:args.num_classes]

    # Split work among threads
    all_images = []
    processing_queue = [(class_id, anp) for class_id, anp in enumerate(unseen_anps)]
    chunks_generator = chunks(processing_queue, int(len(processing_queue) / args.n_threads))
    lock = Lock()

    # Launch threads to list the files
    print('Launching processes (2)')
    sys.stdout.flush()
    processes = [Process(target=select_random_samples, args=(i, chunks_generator.next(), args, all_images, lock))
                 for i in range(args.n_threads)]
    for p in processes:
        p.start()

    # Wait until all the threads finish
    for p in processes:
        p.join()

    # Write to file
    print('Writing to file')
    sys.stdout.flush()
    with open(os.path.join(args.output_path, '%s_image_list.txt' % args.prefix), 'w') as f:
        f.writelines(all_images)
    with open(os.path.join(args.output_path, '%s_class_list.txt' % args.prefix), 'w') as f:
        f.writelines(['%s\n' % anp for anp in unseen_anps])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create list of ANPs for zero-shot learning task')
    parser.add_argument('--all_anps', help='Text file with the list of all possible ANPs')
    parser.add_argument('--used_anps', help='Text file with the list of ANPs used for training')
    parser.add_argument('--num_samples', type=int, help='Minimum samples per class')
    parser.add_argument('--num_classes', type=int, help='Number of classes for the zero-shot experiment')
    parser.add_argument('--mvso_images_dir', help='Directory with class/images.jpg for MVSO')
    parser.add_argument('--output_path', help='Directory where the output files will be stored')
    parser.add_argument('--prefix', default='zero_shot_%d' % random.randint(0, 100),
                        help='Prefix for the output file names')
    parser.add_argument('--n_threads', type=int, default=1, help='Number of threads')

    arguments = parser.parse_args()

    main(arguments)
