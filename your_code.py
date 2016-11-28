import logging
import random

from scoring import calculate_score
from utils import log_header
from retriever import train_retriever, retrieve_similar_images


def train(args):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that
    is required for testing the model must be saved to file so that the test procedure
    can load, execute and report.

    :param args: Run-time arguments
    """

    log_header('Training network')

    train_retriever(args)


def test(label_dict, image_ids, args):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned.

    :param queries: list of image-IDs. Each element is assumed to be an entry in the
                    test set. Hence, the image with id <id> is located on my computer at
                    './test/pics/<id>.jpg'. Make sure this is the file you work with.
    :param args: Run-time arguments
    :return: A dictionary with keys equal to the images in the queries list,
             and values a list of image-IDs retrieved for that input
    """

    log_header('Starting image retrieval preparations')
    logging.info('Generating random test queries')

    queries = []

    # Generate random queries, just to run the "test"-function.
    # These are elements from the TEST-SET folder
    for i in range(1000):
        queries.append(image_ids[random.randint(0, len(image_ids) - 1)])

    return calculate_score(
        label_dict, queries, retrieve_similar_images(queries, '%s/pics' % args.test_path, args)
    )
