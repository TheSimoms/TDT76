import logging

from utils import log_header, get_number_of_classes, get_number_of_images
from classifier import retrieve_similar_images
from labeler import train_labeler
from retriever import train_retriever


def train(args):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that
    is required for testing the model must be saved to file so that the test procedure
    can load, execute and report.

    :param args: Run-time arguments
    """

    log_header('Training network')

    number_of_classes = get_number_of_classes(args.train_path)
    number_of_images = get_number_of_images(args.train_path)

    train_labeler(number_of_classes, args)
    train_retriever(number_of_classes, number_of_images, args)


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
        results[image_id] = retrieve_similar_images(image_id, '%s/pics' % args.test_path, args)

        logging.info('%s: %s' % (image_id, results[image_id]))

    return results
