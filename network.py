from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.training import saver as tf_saver

from utils import preprocess_image


slim = tf.contrib.slim


def retrieve_similar_images(image_path, args):
    """
    Return similar images for query image.

    :param image_path: Path to query image
    :param args: Run-time arguments
    :return: List of images similar to query image
    """

    bottleneck = classify_image(image_path, args)

    if bottleneck is None:
        return []

    return []


def classify_image(image_path, args):
    """
    Classify image.

    :param image_path: Path to query image
    :param args: Run-time arguments
    :return: Bottleneck layer values for query image
    """

    if not os.path.exists(args.checkpoint):
        logging.critical('Checkpoint %s does not exist' % args.checkpoint)

        sys.exit(1)

    g = tf.Graph()

    with g.as_default():
        input_image = preprocess_image(image_path)

        if input_image is None:
            return

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                input_image, num_classes=6012, is_training=False
            )

        predictions = end_points['PreLogits']

        saver = tf_saver.Saver()
        sess = tf.Session()
        saver.restore(sess, args.checkpoint)

        # Run the evaluation on the image
        return np.squeeze(sess.run(predictions))
