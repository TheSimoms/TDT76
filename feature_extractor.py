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


def feature_network(args):
    """
    Set up feature extractor Tensorflow network

    :param args: Run-time arguments
    :return: Tensorflow network
    """

    return setup_convolutional_network(
        (args.image_size ** 2) * args.number_of_channels,
        get_number_of_labels(generate_dict_from_directory(args.train_path), args), args
    )


def compute_features(images, args):
    """
    Compute feature values for images

    :param images: List of images to compute features for
    :param args: Run-time arguments
    :return: Feature layer values for images
    """

    # Verify that pre-trained feature model exist
    if not os.path.exists(args.feature_model):
        logging.critical('Checkpoint %s does not exist' % args.feature_model)

        sys.exit(1)

    with tf.Graph().as_default():
        # Pre-process images
        input_images = [
            preprocess_image('%s/%s.jpg' % (path, image_id), args) for path, _, image_id in images
        ]

        # Run pre-processed images through network
        return run_network(
            feature_network(args), args.feature_model, args, train=False, value=input_images
        )


def generate_features(args):
    """
    Generate and save feature values for all images in training path

    :param args: Run-time arguments
    """

    log_header('Generating features')

    images = []

    # Add all images in training path
    for image_path, _, image_id in get_images_in_path(args.train_path):
        images.append((image_path, image_id))

    # Generate features for all images
    run_network(
        feature_network(args), args.feature_model, args,
        train=False, testing_data=images, save_path=args.features
    )


def get_features(images, args):
    """
    Fetch or generate feature values for images

    :param images: Images to fetch feature values for
    :param args: Run-time arguments
    :return: List of feature values for images
    """

    return compute_features(
        (('%s/pics' % args.test_path, None, image_id) for image_id in images), args
    )


def generate_training_batch(**kwargs):
    """
    Generate batch of training data to use when training the feature extractor

    :param kwargs: Keyword arguments
    :return: Input values and corresponding expected output values for the training batch
    """

    label_dict = kwargs.get('label_dict')
    args = kwargs.get('args')

    # Fetch all labels in training path and set up a list with one zero value for each label
    labels = get_sorted_labels(label_dict, args)
    label_list = [0.0] * len(labels)

    # Fetch a random sample of images from the training path
    images = get_random_sample_of_images_in_path(args.train_path, label_dict, args)

    logging.debug('Generating batch')

    inputs = []
    outputs = []

    # Generate input and output values for each image
    for image_path, _, image_id in images:
        output = list(label_list)

        # Set values corresponding labels in image to its respective confidence
        for label, confidence in label_dict[image_id]:
            if label in labels:
                output[labels.index(label)] = confidence

        inputs.append(preprocess_image('%s/%s.jpg' % (image_path, image_id), args))
        outputs.append(output)

    return inputs, outputs


def train_feature_model(label_dict, args):
    """
    Train the feature extractor

    :param label_dict: Dictionary containing labels (and their confidence)
    :param args: Run-time arguments
    """

    log_header('Training feature model')

    run_network(
        feature_network(args), args.feature_model, args, training_data=(
            generate_training_batch, {'label_dict': label_dict, 'args': args}
        )
    )
