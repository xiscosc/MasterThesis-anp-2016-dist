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


def split_in_classes(input_list, anp_to_anp_id, anp_to_noun_id, anp_to_adj_id):
    """
    Generates a dictionary with a list of (path, label) lines for each label
    :param input_list: lines read from input file
    :param anp_to_anp_id: ANP -> ANP_id mapping
    :param anp_to_noun_id: ANP -> Noun_id mapping
    :param anp_to_adj_id: ANP -> Adj_id mapping
    :return: dictionary with the classes
    """
    classes_dict = dict()
    for elem in input_list:
        split_path = elem.split()[0].split('/')
        anp, img_name = split_path[-2], split_path[-1]
        if anp not in classes_dict.keys():
            classes_dict[anp] = list()
        classes_dict[anp].append('%s/%s %d %d %d' % (anp, img_name, anp_to_anp_id[anp], anp_to_noun_id[anp],
                                                     anp_to_adj_id[anp]))
    return classes_dict


def dict_to_list(input_dict):
    ret_list = list()
    for k in input_dict.keys(): ret_list.extend(input_dict[k])
    return ret_list


def create_mappings_from_file(anp_file, noun_file, adj_file):
    """
    Creates dictionaries that map from anp_name to anp_id, noun_id, adj_id
    :param anp_file:
    :param noun_file:
    :param adj_file:
    :return: anp_to_anp_id, anp_to_noun_id, adj_to_noun_id
    """
    # Create class_name->id mappings for ANPs, Nouns and Adjs
    anp_list = [line.strip() for line in open(anp_file, 'r')]
    noun_list = [line.strip() for line in open(noun_file, 'r')]
    adj_list = [line.strip() for line in open(adj_file, 'r')]
    anp_to_anp_id = dict()
    noun_ids = dict()
    adj_ids = dict()
    for i, anp in enumerate(anp_list): anp_to_anp_id[anp] = i
    for i, noun in enumerate(noun_list): noun_ids[noun] = i
    for i, adj in enumerate(adj_list): adj_ids[adj] = i
    # Create ANP->Noun_id and ANP->Adj_id mappings
    anp_to_noun_id = dict()
    anp_to_adj_id = dict()
    for anp in anp_to_anp_id.keys():
        adj, noun = anp.split('_')
        anp_to_noun_id[anp] = noun_ids[noun]
        anp_to_adj_id[anp] = adj_ids[adj]
    return anp_to_anp_id, anp_to_noun_id, anp_to_adj_id


def create_mappings(file_list):
    """
    Creates dictionaries that map from anp_name to anp_id, noun_id, adj_id
    :param file_list: list read from the paths file
    :return: anp_to_anp_id, anp_to_noun_id, adj_to_noun_id
    """
    # Create list of ANPs, Nouns and Adjectives
    anp_list = list()
    noun_list = list()
    adj_list = list()
    for line in file_list:
        anp = line.split()[0].split('/')[-2]
        adj, noun = anp.split('_')
        if anp not in anp_list: anp_list.append(anp)
        if noun not in noun_list: noun_list.append(noun)
        if adj not in adj_list: adj_list.append(adj)

    # Create mappings
    anp_to_anp_id = dict()
    noun_ids = dict()
    adj_ids = dict()
    for i, anp in enumerate(anp_list): anp_to_anp_id[anp] = i
    for i, noun in enumerate(noun_list): noun_ids[noun] = i
    for i, adj in enumerate(adj_list): adj_ids[adj] = i

    # Create ANP->Noun_id and ANP->Adj_id mappings
    anp_to_noun_id = dict()
    anp_to_adj_id = dict()
    for anp in anp_to_anp_id.keys():
        adj, noun = anp.split('_')
        anp_to_noun_id[anp] = noun_ids[noun]
        anp_to_adj_id[anp] = adj_ids[adj]

    return anp_to_anp_id, anp_to_noun_id, anp_to_adj_id, anp_list, noun_list, adj_list


class Verbose:
    def __init__(self, start_text, done_text=' Done'):
        self.start_text = start_text
        self.done_text = done_text

    def __enter__(self):
        sys.stdout.write(self.start_text)
        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        print self.done_text


def main(args):
    # Read file
    with Verbose('Parsing file...'):
        lines = [line.strip() for line in open(args.input_file_path, 'r')]

    # Create class->id mappings
    with Verbose('Creating mappings... '):
        # mappings = create_mappings_from_file(args.anp_list_path, args.noun_list_path, args.adj_list_path)
        mappings = create_mappings(lines)

    # Create dictionaries containing lists of (path, label) for each class
    with Verbose('Listing all files... '):
        class_dict = split_in_classes(lines, *mappings[0:3])

    with Verbose('Separating classes... '):
        train_dict = dict()
        val_dict = dict()
        for k in class_dict.keys():
            train_dict[k], val_dict[k] = split_list(class_dict[k], args.train_split)

    # Convert dictionaries to lists
    train_list = dict_to_list(train_dict)
    val_list = dict_to_list(val_dict)

    # Shuffle if needed
    if args.shuffle:
        with Verbose('Shuffling... '):
            random.shuffle(train_list)
            random.shuffle(val_list)

    # Save lists to text files
    with Verbose('Writing files... '):
        with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
            f.writelines("%s\n" % line for line in train_list)
        with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
            f.writelines("%s\n" % line for line in val_list)
        with open(os.path.join(args.output_dir, 'anp_list.txt'), 'w') as f:
            f.writelines("%s\n" % line for line in mappings[3])
        with open(os.path.join(args.output_dir, 'noun_list.txt'), 'w') as f:
            f.writelines("%s\n" % line for line in mappings[4])
        with open(os.path.join(args.output_dir, 'adj_list.txt'), 'w') as f:
            f.writelines("%s\n" % line for line in mappings[5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates random stratified train/val splits for MVSO-EN')
    parser.add_argument('--train_split', dest='train_split', type=float,
                        help='Fraction of the dataset used for training')
    parser.add_argument('--input_file', dest='input_file_path',
                        help='Text file with (path, ANP_label) pairs for the whole dataset')
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory where the generated train.txt and val.txt files will be stored')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
                        help='Shuffle the path lists')
    '''
    parser.add_argument('--anp_list', dest='anp_list_path',
                        help='Text file listing the ANPs in the dataset')
    parser.add_argument('--noun_list', dest='noun_list_path',
                        help='Text file listing the nouns in the dataset')
    parser.add_argument('--adj_list', dest='adj_list_path',
                        help='Text file listing the adjectives in the dataset')
    '''
    arguments = parser.parse_args()
    main(arguments)
