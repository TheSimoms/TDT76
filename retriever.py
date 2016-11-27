import logging

from utils import (
    log_header, get_number_of_images, read_pickle, get_sorted_image_ids, get_number_of_labels,
    generate_dict_from_directory
)
from network import Layer, setup_network, run_network
from feature_extractor import get_features, generate_features


def generate_training_data_set(path, args):
    """
    Generate data set used for training the model.

    :param path: Path containing the data
    :return: Data set in format [[input][output]]
    """

    logging.debug('Generating retriever training data set')

    inputs = []
    outputs = []
    count = [0] * get_number_of_images(path)

    features = read_pickle(args.features, False)

    if features is None:
        features = generate_features(args.train_path)

    image_ids = get_sorted_image_ids(path)

    for i in range(len(image_ids)):
        image_id = image_ids[i]

        output = list(count)
        output[i] = 1.0

        inputs.append(features[image_id])
        outputs.append(output)

    return inputs, outputs


def train_retriever(args):
    log_header('Training retriever')

    network = setup_network(
        get_number_of_labels(generate_dict_from_directory(args.train_path)),
        get_number_of_images(args.train_path), [Layer(2048), Layer(4096)], args
    )

    run_network(
        network, args.retrieval_model, args,
        training_data=generate_training_data_set(args.train_path, args)
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
