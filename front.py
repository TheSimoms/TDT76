import random
import argparse
import logging

from utils import log_header, generate_dict_from_directory
from your_code import train, test


def score(label_dict, target, selection, n=50):
    """
    Calculate the score of a selected set compared to the target image.

    :param label_dict: Dictionary of labels, keys are image IDs
    :param target: Image ID of the query image
    :param selection: The list of IDs retrieved
    :param n: The assumed number of relevant images. Kept fixed at 50
    :return: The calculated score
    """

    # Remove the queried element
    selection = list(set(selection) - {target})

    # Fetch labels for queried image
    if target in label_dict.keys():
        target_dict = dict(label_dict[target])
    else:
        logging.error('Could not find %s in the dict keys' % target)

        target_dict = {}

    # Current score will accumulate the element-wise scores, before rescaling by scaling by 2/(k*n)
    current_score = 0.0

    # Calculate best possible score of image
    best_score = sum(target_dict.values())

    # Avoid problems with div zero. If best_score is 0.0 we will
    # get 0.0 anyway, then best_score makes no difference
    if best_score == 0.0:
        best_score = 1.0

    # Loop through all the selected elements
    for selected_element in selection:
        # If we have added a non-existing image we will not get
        # anything, and create a dict with no elements
        # Otherwise select the current labels
        if selected_element in label_dict.keys():
            selected_dict = dict(label_dict[selected_element])
        else:
            logging.error('Could not find %s in the dict keys' % selected_element)

            selected_dict = {}

        # Extract the shared elements
        common_elements = list(set(selected_dict.keys()) & set(target_dict.keys()))

        if len(common_elements) > 0:
            # for each shared element, the potential score is the
            # level of certainty in the element for each of the
            # images, multiplied together
            element_scores = [
                selected_dict[element] * target_dict[element] for element in common_elements
            ]

            # We sum the contributions, and that's it
            current_score += sum(element_scores) / best_score
        else:
            # If there are no shared elements,
            # we won't add anything
            pass

    # We are done after scaling
    return current_score * 2 / (len(selection) + n)


def calculate_score(label_dict, queries, results):
    """
    Calculate score for queried images.

    :param label_dict: Dictionary of labels, keys are image IDs
    :param queries: List of image ids to query
    :param results: Retrieved image ids for each image in queries
    :return: Total score
    """

    total_score = 0.0

    log_header('Individual image scores')

    # Calculate score for all images
    for image_id in queries:
        if image_id in results.keys():
            # Run the score function
            image_score = score(
                label_dict=label_dict, target=image_id, selection=results[image_id]
            )
        else:
            logging.error('No result generated for %s' % image_id)

            image_score = 0.0

        total_score += image_score

        logging.info('%s: %8.6f' % (image_id, image_score))

    log_header('Average score over %d images: %10.8f' % (
        len(queries), total_score / len(queries)
    ))

    return total_score


def run_test(label_dict, image_ids, args):
    """
    Run the test procedure.

    :param label_dict: Dictionary of labels, keys are image IDs
    :param image_ids: List of image ids to query
    :param args: Run-time arguments
    :return: Total score
    """

    queries = []

    logging.info('Generating random test queries')

    # Generate random queries, just to run the "test"-function.
    # These are elements from the TEST-SET folder
    for i in range(1000):
        queries.append(image_ids[random.randint(0, len(image_ids) - 1)])

    # Calculate the score
    return calculate_score(label_dict, queries, test(queries, args))


def main():
    """
    Run the test procedure, with option to train the network first.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Train the network before testing')
    parser.add_argument('--debug', action='store_true', help='Toggle debug')
    parser.add_argument(
        '--checkpoint', type=str, default='./models/2016_08/model.ckpt',
        help='Path to optional custom pre-trained model'
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Make sure we have generated a list of train IDS and their labels stored in a pickle
    train_labels = generate_dict_from_directory()

    # Optionally train the network
    if args.train:
        logging.info('Training network')

        train(args)

    # Make sure we have generated a list of test IDS and their labels stored in a pickle
    test_labels = generate_dict_from_directory(
        pickle_file='./test/pickle/combined.pickle', directory='./test/txt/'
    )

    # Run the testing
    run_test({**test_labels, **train_labels}, list(test_labels.keys()), args)


if __name__ == "__main__":
    main()
