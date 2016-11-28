from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import pickle
import glob
import ntpath
import random
import numpy as np

from collections import defaultdict
from PIL import Image


SEPARATOR = 50 * '='


def generate_dict_from_directory(path):
    """
    Go through a full directory of .txt-files, and transform the .txt into dict.
    Create a combo of all the txt-files which is saved in a pickle.
    Also save a small .pickle per txt-file

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
    (label, belief) - tuples as value

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
    Return path name, file name and image ID for all image files in path

    :param path: Path to search
    :return: List of tuples (path, filename, image ID) for each image in path
    """

    logging.debug('Fetching images in path %s' % path)

    images = []

    for full_path in glob.glob(path + '/pics/*/*.jpg'):
        path, filename = ntpath.split(full_path)

        images.append((path, filename, filename.split('.')[0]))

    return images


def get_random_sample_of_images_in_path(path, label_dict, args):
    """
    Get random sample of images in path

    :param header: Path to images
    :param label_dict: Dictionary containing labels for images
    :return: List of tuples (path, filename, filename) for each image in path
    """

    # Get
    images = get_images_in_path(path)
    res = set()

    while len(res) < args.batch_size:
        image = _, _, image_id = random.choice(images)

        if image_id in label_dict:
            res.add(image)

    return res


def get_sorted_labels(label_dict, args):
    """
    Get sorted list of labels

    :param label_dict: Dictionary containing labels for images
    :return: Sorted list of labels
    """

    number_of_images = len(label_dict)

    # List of labels
    label_list = list()

    # Dictionary containing number of occurrences for each label
    counts = defaultdict(int)

    # Iterate all images, update label counts
    for _, labels in label_dict.items():
        for label, _ in labels:
            counts[label] += 1

    # Iterate all labels, remove labels that are too rare. This is done to speed up computations
    for label in counts:
        usage = counts[label] / number_of_images * 100

        if usage > args.label_threshold:
            label_list.append(label)

    return sorted(label_list)


def get_number_of_labels(label_dict, args):
    """
    Get the number of different labels

    :param label_dict: Dictionary containing labels for images
    :return: Number of labels
    """

    return len(get_sorted_labels(label_dict, args))


def get_sorted_image_ids(path):
    """
    Get sorted list of image IDs in path

    :param header: Path to images
    :return: Sorted list of image IDs
    """

    return sorted(generate_dict_from_directory(path).keys())


def get_number_of_images(path):
    """
    Get the number of images in path

    :param header: Path to images
    :return: Number of images
    """

    return len(generate_dict_from_directory(path))


def log_header(header):
    """
    Display header, separated by lines of equality symbols

    :param header: Header to display
    """

    logging.info(SEPARATOR)
    logging.info(header)
    logging.info(SEPARATOR)


def read_pickle(filename, vital=True):
    """
    Read pickle file

    :param filename: Filename to save value as
    :param vital: Whether to exit the program when file not found
    :return: Value or None if file not found
    """

    if not os.path.exists(filename):
        if vital:
            logging.critical('Vital file %s does not exist. Aborting' % filename)

            sys.exit(1)

        return None

    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle(value, filename):
    """
    Save value to pickle file

    :param value: Value to save
    :param filename: Filename to save value as
    """

    with open(filename, 'wb') as f:
        return pickle.dump(value, f)


def preprocess_image(image_path, args):
    """
    Load and preprocess an image

    :param image_path: Path to an image
    :param args: Run-time arguments
    :return: An ops.Tensor that produces the preprocessed image.
    """

    logging.debug('Preprocessing image %s' % image_path)

    if not os.path.exists(image_path):
        logging.error('Input image does not exist %s' % image_path)

        return

    # Load image
    try:
        img = Image.open(image_path)
    except OSError:
        return np.zeros(args.image_size * args.image_size * args.number_of_channels)

    # Resize image to (args.image_size, args.image_size)
    img = img.resize((args.image_size, args.image_size), Image.ANTIALIAS)

    # Convert image to numpy array
    img = np.array(img, dtype='float32')

    # Flatten image
    img = img.ravel()

    return img
