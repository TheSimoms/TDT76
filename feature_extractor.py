from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import tensorflow as tf

from utils import (
    preprocess_image, get_images_in_path, generate_dict_from_directory, get_sorted_labels,
    get_number_of_labels, log_header, get_random_sample_of_images_in_path
)
from network import (
    setup_convolutional_network, run_network
)


def compute_features(images, args):
    """
    Compute feature values for image.

    :param image_id: ID of query image
    :param args: Run-time arguments
    :return: Feature layer values for query image
    """

    if not os.path.exists(args.feature_model):
        logging.critical('Checkpoint %s does not exist' % args.feature_model)

        sys.exit(1)

    with tf.Graph().as_default():
        input_images = [
            preprocess_image('%s/%s.jpg' % (path, image_id), args) for path, _, image_id in images
        ]

        network = setup_convolutional_network(
            (args.image_size ** 2) * args.number_of_channels,
            get_number_of_labels(generate_dict_from_directory(args.train_path), args), args
        )

        return run_network(
            network, args.feature_model, args, train=False, value=input_images
        )


def generate_features(args):
    """
    Generate and save feature values for all images in given paths.

    :param paths: Paths containing the images
    :return: Dict <K: image ID, V: features>
    """

    log_header('Generating features')

    images = []

    for image_path, _, image_id in get_images_in_path(args.train_path):
        images.append((image_path, image_id))

    network = setup_convolutional_network(
        (args.image_size ** 2) * args.number_of_channels,
        get_number_of_labels(generate_dict_from_directory(args.train_path), args), args
    )

    run_network(
        network, args.feature_model, args, train=False, testing_data=images,
        save_path=args.features
    )


def get_features(images, path, args):
    """
    Fetch or generate feature values for image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: Features
    """

    return compute_features(((path, None, image_id) for image_id in images), args)


def generate_training_batch(**kwargs):
    label_dict = kwargs.get('label_dict')
    args = kwargs.get('args')

    labels = get_sorted_labels(label_dict, args)
    images = get_random_sample_of_images_in_path(args.train_path, label_dict, args)

    label_list = [0.0] * len(labels)

    logging.debug('Generating batch')

    inputs = []
    outputs = []

    for image_path, _, image_id in images:
        output = list(label_list)

        for label, confidence in label_dict[image_id]:
            if label in labels:
                output[labels.index(label)] = confidence

        inputs.append(preprocess_image('%s/%s.jpg' % (image_path, image_id), args))
        outputs.append(output)

    return inputs, outputs


def train_feature_model(label_dict, args):
    """
    Train feature extractor model.
    """

    log_header('Training feature model')

    network = setup_convolutional_network(
        (args.image_size ** 2) * args.number_of_channels, get_number_of_labels(label_dict, args),
        args
    )

    run_network(
        network, args.feature_model, args, training_data=(
            generate_training_batch, {'label_dict': label_dict, 'args': args}
        )
    )
