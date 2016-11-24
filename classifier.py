from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.training import saver as tf_saver

from utils import preprocess_image, get_images_in_path, generate_dict_from_directory


slim = tf.contrib.slim


def compute_bottleneck(image_id, path, args):
    """
    Compute bottleneck values for image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: Bottleneck layer values for query image
    """

    if not os.path.exists(args.image_model):
        logging.critical('Checkpoint %s does not exist' % args.image_model)

        sys.exit(1)

    logging.debug('Computing bottleneck values for image %s' % image_id)

    g = tf.Graph()

    with g.as_default():
        input_image = preprocess_image('%s/%s.jpg' % (path, image_id))

        if input_image is None:
            return

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                input_image, num_classes=6012, is_training=False
            )

        predictions = end_points['PreLogits']

        saver = tf_saver.Saver()
        sess = tf.Session()
        saver.restore(sess, args.image_model)

        return np.squeeze(sess.run(predictions))


def generate_bottlenecks(paths, args):
    """
    Generate and save bottleneck values for all images in given paths.

    :param paths: Paths containing the images
    :return: Dict <K: image ID, V: bottleneck>
    """

    bottlenecks = {}

    for path in paths:
        logging.debug('Generating bottleneck values for path %s' % path)

        label_dict = generate_dict_from_directory(path)

        for image_path, _, image_id in get_images_in_path(path):
            if image_id not in label_dict:
                continue

            bottlenecks[image_id] = compute_bottleneck(image_id, image_path, args)

    with open(args.bottleneck, 'wb') as f:
        pickle.dump(bottlenecks, f)

        logging.info('Bottleneck values saved to file %s' % args.bottleneck)

    return bottlenecks


def get_bottleneck(image_id, path, args):
    """
    Fetch or generate bottleneck values for image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: Bottleneck
    """

    if os.path.isfile(args.bottleneck):
        with open(args.bottleneck, 'rb') as f:
            return pickle.load(f)[image_id]
    else:
        return compute_bottleneck(image_id, path, args)


def run_bottleneck_through_network(bottleneck, args):
    """
    Run bottleneck values through network.

    :param bottleneck: Bottleneck values for image
    :param args: Run-time arguments
    :return: Output layer
    """

    if not os.path.isfile(args.retrieval_model):
        logging.error('Could not find pre-trained classifier model. Please train a model frist')

        sys.exit(1)

    print(len(bottleneck))

    return None


def retrieve_similar_images(image_id, path, args):
    """
    Return similar images for query image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: List of images similar to query image
    """

    bottleneck = get_bottleneck(image_id, path, args)

    if bottleneck is None:
        return []

    return []
