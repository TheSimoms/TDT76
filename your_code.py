import logging

from utils import log_header, get_sorted_image_ids
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


def test(queries, args):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned.

    :param queries: list of image-IDs. Each element is assumed to be an entry in the
                    test set. Hence, the image with id <id> is located on my computer at
                    './test/pics/<id>.jpg'. Make sure this is the file you work with.
    :param args: Run-time arguments
    :return: A dictionary with keys equal to the images in the queries list,
             and values a list of image-IDs retrieved for that input
    """

    log_header('Retrieving images')

    results = {}

    for image_id in queries:
        results[image_id] = retrieve_similar_images(
            image_id, '%s/pics' % args.test_path, get_sorted_image_ids(args.train_path), args
        )

        logging.info('%s: %s' % (image_id, results[image_id]))

    return results
