import logging
import random

from utils import (
    log_header, get_number_of_images, read_pickle, get_number_of_labels,
    generate_dict_from_directory
)
from network import Layer, setup_network, run_network
from feature_extractor import get_features, generate_features


def generate_training_batch(**kwargs):
    """
    Generate data set used for training the model.

    :param path: Path containing the data
    :return: Data set in format [[input][output]]
    """

    logging.debug('Generating batch')

    label_dict = kwargs.get('label_dict')
    args = kwargs.get('args')

    inputs = []
    outputs = []

    count = [0.0] * len(label_dict)

    features = read_pickle(args.features, False)

    if features is None:
        features = generate_features(args.train_path)

    image_ids = sorted(label_dict.keys())
    image_set = set()

    while len(image_set) < args.batch_size:
        image_set.add(random.choice(image_ids))

    for image_id in image_set:
        i = image_ids.index(image_id)

        output = list(count)
        output[i] = 1.0

        inputs.append(features[image_id])
        outputs.append(output)

    return inputs, outputs


def train_retriever(label_dict, args):
    log_header('Training retriever')

    network = setup_network(
        get_number_of_labels(generate_dict_from_directory(args.train_path)),
        get_number_of_images(args.train_path), [Layer(2048), Layer(4096)], args
    )

    run_network(
        network, args.retrieval_model, args, training_data=(
            generate_training_batch, {'label_dict': label_dict, 'args': args}
        ),
    )


def retrieve_similar_images(image_id, path, image_ids, args):
    """
    Return similar images for query image.

    :param image_id: ID of query image
    :param path: Path where image is located
    :param args: Run-time arguments
    :return: List of images similar to query image
    """

    network = setup_network(
        get_number_of_labels(generate_dict_from_directory(args.train_path)),
        get_number_of_images(args.train_path), [Layer(2048), Layer(4096)], args
    )

    output_layer = run_network(
        network, args.retrieval_model, args,
        train=False, value=get_features(image_id, path, args)
    )

    print(output_layer)

    return list(
        image_ids[i] for i in range(len(output_layer)) if output_layer[i] >= args.threshold
    )
