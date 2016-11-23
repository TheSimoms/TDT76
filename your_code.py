import logging

from utils import log_header
from network import retrieve_similar_images


def train(args, location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that
    is required for testing the model must be saved to file so that the test procedure
    can load, execute and report.

    :param location: The location of the training data folder hierarchy
    """

    pass


def test(queries, args, location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned.

    :param queries: list of image-IDs. Each element is assumed to be an entry in the
                    test set. Hence, the image with id <id> is located on my computer at
                    './test/pics/<id>.jpg'. Make sure this is the file you work with.
    :param location: The location of the test data folder hierarchy
    :return: A dictionary with keys equal to the images in the queries list,
             and values a list of image-IDs retrieved for that input
    """

    results = {}

    log_header('Retrieving images')

    for image_id in queries:
        results[image_id] = retrieve_similar_images(
            '%s/pics/%s.jpg' % (location, image_id), args
        )

        logging.info('%s: %s' % (image_id, results[image_id]))

    return results
