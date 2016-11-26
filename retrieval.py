import logging

from utils import log_header, get_number_of_images, read_pickle, get_sorted_image_ids
from network import Layer, network
from classifier import get_bottleneck


def generate_training_data_set(path, args):
    """
    Generate data set used for training the model.

    :param path: Path containing the data
    :return: Data set in format [[input][output]]
    """

    logging.debug('Generating retriever training data set')

    count = [0] * get_number_of_images(path)
    inputs = []
    outputs = []

    bottlenecks = read_pickle(args.bottlenecks)
    image_ids = get_sorted_image_ids(path)

    for i in range(len(image_ids)):
        image_id = image_ids[i]

        output = list(count)
        output[i] = 1.0

        inputs.append(bottlenecks[image_id])
        outputs.append(output)

    return inputs, outputs


def train_retriever(args):
    log_header('Training retriever')

    network(
        [Layer(2048), Layer(4096)], args.retrieval_model, args,
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

    output_layer, last_layer = network(
        [Layer(2048), Layer(4096)], args.retrieval_model, args,
        train=False, value=get_bottleneck(image_id, path, args)
    )

    return list(
        image_ids[i] for i in range(len(output_layer)) if output_layer[i] >= args.threshold
    )
