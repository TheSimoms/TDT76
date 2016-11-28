import logging
import random
import glob
import numpy as np

from utils import (
    get_number_of_images, read_pickle, get_number_of_labels,
    generate_dict_from_directory, get_sorted_image_ids
)
from network import Layer, setup_network, run_network
from feature_extractor import get_features


def generate_training_batch(**kwargs):
    """
    Generate data set used for training the model.

    :param path: Path containing the data
    :return: Data set in format [[input][output]]
    """

    logging.debug('Generating batch')

    args = kwargs.get('args')

    inputs = []
    outputs = []

    image_ids = get_sorted_image_ids(args.train_path)
    count = [0.0] * len(image_ids)

    features = None
    feature_files = glob.glob('%s.*' % args.features)

    while features is None:
        filename = random.choice(feature_files)
        features = read_pickle(filename, False)

    feature_batch_keys = random.sample(features.keys(), args.batch_size)

    for image_id in feature_batch_keys:
        if image_id in image_ids:
            output = list(count)
            output[image_ids.index(image_id)] = 1.0

            inputs.append(features[image_id])
            outputs.append(output)

    return inputs, outputs


def train_retriever(args):
    network = setup_network(
        get_number_of_labels(generate_dict_from_directory(args.train_path), args),
        get_number_of_images(args.train_path), [Layer(512), Layer(512)], args
    )

    run_network(
        network, args.retrieval_model, args, training_data=(
            generate_training_batch, {'args': args}
        ),
    )


def retrieve_similar_images(queries, path, args):
    """
    Return similar images for query image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: List of images similar to query image
    """

    image_ids = get_sorted_image_ids(args.train_path)

    logging.info('Generating image features')

    features = get_features(queries, path, args)

    network = setup_network(
        get_number_of_labels(generate_dict_from_directory(args.train_path), args),
        get_number_of_images(args.train_path), [Layer(512), Layer(512)], args
    )

    logging.info('Calculating image similarities')

    output_layers = run_network(
        network, args.retrieval_model, args,
        train=False, value=features
    )

    res = {}

    for i in range(len(output_layers)):
        image_id = queries[i]

        retrieved = output_layers[i]
        min_value = np.amin(retrieved)

        if min_value < 0:
            retrieved -= np.amin(retrieved)

        retrieved /= np.amax(retrieved)
        top_indices = retrieved.argsort()[::-1][:50]

        res[image_id] = list(
            image_ids[j] for j in top_indices if retrieved[j] > args.threshold
        )

        logging.debug('%s: %d images retrieved' % (image_id, len(res[image_id])))

    return res
