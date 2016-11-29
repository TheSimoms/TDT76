import logging
import random

from scoring import calculate_score
from utils import log_header
from retriever import train_retriever, retrieve_similar_images


def train(args):
    """
    Run the training procedure

    :param args: Run-time arguments
    """

    log_header('Training network')

    train_retriever(args)


def test(label_dict, image_ids, args):
    """
    Test the retrieval system. For each input in image_ids, generate a list of IDs returned for
    that image.

    :param label_dict: Dictionary containing labels (and their confidence) for each image
    :param image_ids: List of image IDs. Each element is assumed to be an entry in the test set,
                      and is located at "pics/<image id>.jpg" in the test path
    :param args: Run-time arguments
    :return: A dictionary with keys equal to the images in the queries list,
             and values a list of image IDs retrieved for that input
    """

    log_header('Starting image retrieval preparations')

    queries = []

    if args.test_image is not None:
        queries.append(args.test_image)
    else:
        logging.info('Generating random test queries')

        number_of_images = len(image_ids)

        # Generate random queries to use in the test procedure
        for i in range(args.k):
            queries.append(image_ids[random.randint(0, number_of_images - 1)])

    # Calculate score for the retrieved images
    calculate_score(
        label_dict, queries, retrieve_similar_images(queries, args)
    )
