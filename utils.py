from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import pickle
import glob
import ntpath
import numpy as np

from scipy import misc


SEPARATOR = 50 * '='


def generate_dict_from_directory(path):
    """
    Go through a full directory of .txt-files, and transform the .txt into dict.
    Create a combo of all the txt-files which is saved in a pickle.
    Also save a small .pickle per txt-file.

    :param path: Path where data is stored
    :return: A dict
    """

    pickle_file = '%s/pickle/combined.pickle' % path
    directory = '%s/txt/' % path

    if os.path.isfile(pickle_file):
        # This is easy - dict generated already
        with open(pickle_file, 'rb') as f:
            my_dict = pickle.load(f)
    else:
        my_dict = {}

        for f in glob.glob(directory + '/*.txt'):
            # Add elements from dict; Requires Python 3.5
            my_dict.update(generate_dict_from_text_file(f))

        with open(pickle_file, 'wb') as f:
            pickle.dump(my_dict, f)

    return my_dict


def generate_dict_from_text_file(filename):
    """
    The workhorse of the previous def; take a single text file and store its content
    into a dict. The dict is defined using image IDs as keys, and a vector of
    (label, belief) - tuples as value.

    :param filename: Name of the .txt to read
    :return: The dict
    """

    logging.debug('Reading %s' % filename)

    if os.path.isfile(filename):
        my_dict = {}

        with open(filename, 'r') as f:
            for line in f.readlines():
                segments = line.rstrip('\n').split(';')

                val = []

                for my_segment in segments[1:]:
                    # my_segment contains a word/phrase with a score in parenthesis.
                    # Skip element 0, as that is the key.
                    parenthesis = my_segment.rfind('(')

                    if parenthesis > 0:
                        # We found something
                        val.append(
                            tuple(
                                [my_segment[:parenthesis], float(my_segment[parenthesis + 1:-1])])
                            )

                my_dict[segments[0]] = val

        with open(filename.replace('/txt/', '/pickle/').replace('.txt', '.pickle'), 'wb') as f:
            pickle.dump(my_dict, f)
    else:
        logging.error('File does not exist: %s' % filename)

    return my_dict


def get_images_in_path(path):
    """
    Return path name and file name for all image files in path.

    :param path: Path to search
    :return: List of tuples (path, filename)
    """

    logging.debug('Fetching images in path %s' % path)

    images = []

    for full_path in glob.glob(path + '/pics/*/*.jpg'):
        path, filename = ntpath.split(full_path)

        images.append((path, filename, filename.split('.')[0]))

    return images


def get_sorted_labels(label_dict):
    """
    Get the number of different labels.

    :param header: Path to images
    :return: Number of labels
    """

    label_set = set()

    for _, labels in label_dict.items():
        label_set.update(label[0] for label in labels)

    return sorted(label_set)


def get_number_of_labels(label_dict):
    """
    Get the number of different labels.

    :param header: Path to images
    :return: Number of labels
    """

    return len(get_sorted_labels(label_dict))


def get_sorted_image_ids(path):
    return sorted(generate_dict_from_directory(path).keys())


def get_number_of_images(path):
    """
    Get the number of images.

    :param header: Path to images
    :return: Number of images
    """

    return len(generate_dict_from_directory(path))


def log_header(header):
    """
    Display header, separated by lines of equality symbols.

    :param header: Header to display
    """

    logging.info(SEPARATOR)
    logging.info(header)
    logging.info(SEPARATOR)


def read_pickle(filename, vital=True):
    if not os.path.exists(filename):
        if vital:
            logging.critical('Vital file %s does not exist. Aborting' % filename)

            sys.exit(1)

        return None

    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle(value, filename):
    with open(filename, 'wb') as f:
        return pickle.dump(value, f)


def preprocess_image(image_path, central_fraction=0.875):
    """
    Load and preprocess an image.

    :param image_path: Path to an image
    :param central_fraction: Do a central crop with the specified fraction of image covered.
    :return: An ops.Tensor that produces the preprocessed image.
    """

    if not os.path.exists(image_path):
        logging.error('Input image does not exist %s' % image_path)

        return

    img = misc.imread(image_path)

    img = np.array(img, dtype='float32')

    img = img.ravel()

    return img
