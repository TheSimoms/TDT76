import logging
import random
import glob

from utils import (
    get_number_of_images, read_pickle, get_number_of_labels,
    generate_dict_from_directory, get_sorted_image_ids
)
from network import Layer, setup_network, run_network
from feature_extractor import get_features


def retrieval_network(args):
    """
    Set up retrieval Tensorflow network

    :param args: Run-time arguments
    :return: Tensorflow network
    """

    number_of_labels = get_number_of_labels(generate_dict_from_directory(args.train_path), args)

    return setup_network(
        number_of_labels, get_number_of_images(args.train_path),
        [Layer(512), Layer(512)], args
    )


def generate_training_batch(**kwargs):
    """
    Generate batch of training data to use when training the retriever

    :param kwargs: Keyword arguments
    :return: Input values and corresponding expected output values for the training batch
    """

    logging.debug('Generating batch')

    args = kwargs.get('args')

    inputs = []
    outputs = []

    # Get sorted image IDs and set up a list with one zero value for each image
    image_ids = get_sorted_image_ids(args.train_path)
    count = [0.0] * len(image_ids)

    # Placeholder for the features to use in the batch
    features = None

    # List of all feature files
    feature_files = glob.glob('%s.*' % args.features)

    # Fetch a random feature file
    while features is None:
        filename = random.choice(feature_files)
        features = read_pickle(filename, False)

    # Extract a random batch of correct batch size from feature file
    feature_batch_keys = random.sample(features.keys(), args.batch_size)

    # Generate input and output values for each image
    for image_id in feature_batch_keys:
        if image_id in image_ids:
            # Set value corresponding current image to 1.0
            output = list(count)
            output[image_ids.index(image_id)] = 1.0

            inputs.append(features[image_id])
            outputs.append(output)

    return inputs, outputs


def train_retriever(args):
    """
    Train the image retriever

    :param args: Run-time arguments
    """

    run_network(
        retrieval_network(args), args.retrieval_model, args, training_data=(
            generate_training_batch, {'args': args}
        ),
    )


def retrieve_similar_images(query, args):
    """
    Return similar images for query images

    :param query: ID of query image
    :param args: Run-time arguments
    :return: Dictionary with retrieved images for each image ID in the query
    """

    logging.info('Generating image features')

    # Fetch all image IDs in training path
    image_ids = get_sorted_image_ids(args.train_path)

    # Fetch features for the query images
    features = get_features(query, args)

    logging.info('Calculating image similarities')

    # Run the features through the retrieval network
    output_layers = run_network(
        retrieval_network(args), args.retrieval_model, args,
        train=False, value=features
    )

    res = {}

    # Iterate every query image, extracting the most relevant images
    for i in range(len(output_layers)):
        image_id = query[i]

        # Parse network output
        retrieved = output_layers[i]

        top_indices = retrieved.argsort()[::-1][:50]

        # Extract most relevant images
        res[image_id] = list(
            image_ids[j] for j in top_indices if retrieved[j] > args.threshold
        )

        logging.debug('%s: %d images retrieved' % (image_id, len(res[image_id])))

    return res
