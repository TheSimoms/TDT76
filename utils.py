from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import logging
import pickle
import glob
import ntpath
import tensorflow as tf


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
            my_dict = {**generate_dict_from_text_file(f), **my_dict}

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

    images = []

    for full_path in glob.glob(path + '/pics/*/*.jpg'):
        path, filename = ntpath.split(full_path)

        images.append((path, filename, filename.split('.')[0]))

    return images


def log_header(header):
    """
    Display header, separated by lines of equality symbols.

    :param header: Header to display
    """

    logging.info(SEPARATOR)
    logging.info(header)
    logging.info(SEPARATOR)


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

    img_data = tf.gfile.FastGFile(image_path).read()

    # Decode Jpeg data and convert to float.
    img = tf.cast(tf.image.decode_jpeg(img_data, channels=3), tf.float32)

    # Do a central crop
    img = tf.image.central_crop(img, central_fraction=central_fraction)

    # Make into a 4D tensor by setting a 'batch size' of 1.
    img = tf.expand_dims(img, [0])
    img = tf.image.resize_bilinear(img, (299, 299), align_corners=False)

    # Center the image about 128.0 (which is done during training) and normalize.
    img = tf.mul(img, 1.0/127.5)

    return tf.sub(img, 1.0)