from __future__ import print_function

import os
import sys
import urllib2
import argparse
from multiprocessing import Process

parser = argparse.ArgumentParser('Download images from a CSV')
parser.add_argument('--csv_root_dir', help='Directory containing all CSV files')
parser.add_argument('--emotion', help='Which emotion class to download')
parser.add_argument('--output_root_dir', help='Directory where the dataset will be downloaded: output_root_dir/emotion/images')
parser.add_argument('--n_threads', type=int, default=1, help='Number of threads downloading images in parallel')

args = parser.parse_args()

emotion = args.emotion.lower()
download_dir = os.path.join(args.output_root_dir, emotion)
csv_files = [os.path.join(args.csv_root_dir, f) for f in os.listdir(args.csv_root_dir) if f.lower().startswith(emotion)]

# Create download directory if necessary
if not os.path.exists(args.output_root_dir): os.makedirs(args.output_root_dir)
if not os.path.exists(download_dir): os.makedirs(download_dir)

# List all download links
all_download_links = []
for csv_file in csv_files:
    with open(csv_file, 'r') as f:
        all_download_links.extend([line.split(',')[1] for line in f.readlines()])


# Split among threads
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

chunk_size = int(len(all_download_links) / args.n_threads)
download_links_per_thread = [None] * args.n_threads
chunks_generator = chunks(all_download_links, chunk_size)


# Basic image downloading function
def download_list(id, link_list, download_dir):
    for index, url in enumerate(link_list):
        file_path = os.path.join(download_dir, url.split('/')[-1])
        # Avoid downloading already existing files twice
        if os.path.exists(file_path):
            continue
        with open(file_path, "wb") as downloaded_file:
            downloaded_file.write(urllib2.urlopen(url).read())
        if i % 50 == 0:
            print('[Thread #%d] Downloaded %d/%d images' % (id, index+1, len(link_list)))
            sys.stdout.flush()


# Launch threads to download the files
processes = [Process(target=download_list, args=(i, chunks_generator.next(), download_dir))
             for i in range(args.n_threads)]
for p in processes:
    p.start()

# Wait until all the threads finish
for p in processes:
    p.join()
