from __future__ import print_function

import os
import sys
import argparse

parser = argparse.ArgumentParser('Generate text file with (filename, label) pairs for MVSO Sentiment dataset')
parser.add_argument('--csv_file', help='File with the ground truth')
parser.add_argument('--mvso_root_dir', help='Directory containing all MVSO_EN images')
parser.add_argument('--output_file', help='Where the output will be stored')
parser.add_argument('--binary', action='store_true', default=False, help='Whether to use only positve/negative classes')

args = parser.parse_args()


with open(args.csv_file, 'r') as f:
    csv_content = f.readlines()[1:]


missing_images_counter = 0
output_lines = []
for i, line in enumerate(csv_content):
    fields = line.split(',')
    anp = fields[0]
    filename = fields[1].split('/')[-1]
    if not os.path.exists(os.path.join(args.mvso_root_dir, anp, filename)):
        missing_images_counter += 1
        continue
    try:
        sentiment_score = (float(fields[2]) + float(fields[3]) + float(fields[4])) / 3.
    except:
        missing_images_counter += 1
        continue
    if args.binary:
        if sentiment_score < 0.: sentiment_label = 0
        elif sentiment_score > 0.: sentiment_label = 1
        else:
            continue
    else:
        if sentiment_score < -0.5: sentiment_label = 0  # negative: 0
        elif sentiment_score > 0.5: sentiment_label = 2  # positive: 2
        else: sentiment_label = 1  # neutral: 1
    output_lines.append('%s %d\n' % (os.path.join(anp, filename), sentiment_label))
    if i % 10 == 0:
        sys.stdout.write('\rProcessed %d/%d images' % (i, len(csv_content)))
        sys.stdout.flush()

print('\n\nDone!\n')
print('%d/%d images were not found' % (missing_images_counter, len(csv_content)))

with open(args.output_file, 'w') as f:
    f.writelines(output_lines)
