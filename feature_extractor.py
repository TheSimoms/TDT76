from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import pickle
import tensorflow as tf

from utils import (
    preprocess_image, get_images_in_path, generate_dict_from_directory, get_sorted_labels,
    get_number_of_labels, log_header, get_random_sample_of_images_in_path
)
from network import (
    setup_convolutional_network, run_network, IMAGE_SIZE
)


def compute_features(image_id, path, args):
    """
    Compute feature values for image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: Feature layer values for query image
    """

    if not os.path.exists(args.feature_model):
        logging.critical('Checkpoint %s does not exist' % args.feature_model)

        sys.exit(1)

    logging.debug('Computing feature values for image %s' % image_id)

    with tf.Graph().as_default():
        input_image = preprocess_image('%s/%s.jpg' % (path, image_id))

        network = setup_convolutional_network(
            IMAGE_SIZE, get_number_of_labels(generate_dict_from_directory(args.train_path)), args
        )

        return run_network(
            network, args.feature_model, args,
            train=False, value=input_image
        )


def generate_features(paths, args):
    """
    Generate and save feature values for all images in given paths.

    :param paths: Paths containing the images
    :return: Dict <K: image ID, V: features>
    """

    features = {}

    for path in paths:
        logging.debug('Generating feature values for path %s' % path)

        label_dict = generate_dict_from_directory(path)

        for image_path, _, image_id in get_images_in_path(path):
            if image_id not in label_dict:
                continue

            features[image_id] = compute_features(image_id, image_path, args)

    with open(args.features, 'wb') as f:
        pickle.dump(features, f)

        logging.info('Feature values saved to file %s' % args.features)

    return features


def get_features(image_id, path, args):
    """
    Fetch or generate feature values for image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: Features
    """

    if os.path.isfile(args.features):
        with open(args.features, 'rb') as f:
            features = pickle.load(f)

            if image_id in features:
                return features[image_id]

    return compute_features(image_id, path, args)


def generate_training_batch(**kwargs):
    label_dict = kwargs.get('label_dict')
    args = kwargs.get('args')

    labels = get_sorted_labels(label_dict)
    images = get_random_sample_of_images_in_path(args.train_path, label_dict, args)

    label_list = [0.0] * len(labels)

    logging.debug('Generating batch')

    inputs = []
    outputs = []

    for image_path, _, image_id in images:
        output = list(label_list)

        for label, confidence in label_dict[image_id]:
            output[labels.index(label)] = confidence

        inputs.append(preprocess_image('%s/%s.jpg' % (image_path, image_id)))
        outputs.append(output)

    return inputs, outputs


def train_feature_model(label_dict, args):
    """
    Train feature extractor model.
    """

    log_header('Training feature model')

    network = setup_convolutional_network(
        IMAGE_SIZE, get_number_of_labels(label_dict), args
    )

    run_network(
        network, args.feature_model, args, training_data=(
            generate_training_batch, {'label_dict': label_dict, 'args': args}
        )
    )
