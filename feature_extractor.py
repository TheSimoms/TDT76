from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging
import pickle
import math
import tensorflow as tf

from utils import (
    preprocess_image, get_images_in_path, generate_dict_from_directory, get_sorted_labels,
    get_number_of_labels, log_header, read_pickle, save_pickle
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


def generate_training_data_set(label_dict, args):
    training_data = read_pickle(args.training_data, False)

    if training_data is not None:
        return training_data

    labels = get_sorted_labels(label_dict)

    images = get_images_in_path(args.train_path)
    number_of_images = len(images)

    training_data = [[], []]

    logging.debug('Generating training data')

    progress_step = math.ceil(number_of_images / 100)

    for i in range(number_of_images):
        image_path, _, image_id = images[i]

        if i % progress_step == 0:
            logging.info('Progress: %d%%' % (i / number_of_images * 100))

        if image_id not in label_dict:
            continue

        output = []

        for label, confidence in label_dict[image_id]:
            output.append((labels.index(label), confidence))

        training_data[0].append('%s/%s.jpg' % (image_path, image_id))
        training_data[1].append(output)

    save_pickle(training_data, args.training_data)

    return training_data


def train_feature_model(label_dict, args):
    """
    Train feature extractor model.
    """

    log_header('Training feature model')

    training_data = generate_training_data_set(label_dict, args)

    network = setup_convolutional_network(
        IMAGE_SIZE, get_number_of_labels(label_dict), args
    )

    run_network(
        network, args.feature_model, args, training_data=training_data, convolutional=True
    )
