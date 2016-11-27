import sys
import random
import argparse
import logging

from feature_extractor import generate_features, train_feature_model
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

    logging.info('Generating random test queries')

    queries = []

    # Generate random queries, just to run the "test"-function.
    # These are elements from the TEST-SET folder
    # for i in range(1000):
    for i in range(5):
        queries.append(image_ids[random.randint(0, len(image_ids) - 1)])

    queries.append('007904b728ccf6b6')

    # Calculate the score
    return calculate_score(label_dict, queries, test(queries, args))


def main():
    """
    Run the test procedure, with option to train the network first.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store_true', help='Train the network before testing')
    parser.add_argument('--debug', action='store_true', help='Toggle debug')

    parser.add_argument('--train-path', type=str, default='./train', help='Path to training data')
    parser.add_argument('--test-path', type=str, default='./test', help='Path to testing data')
    parser.add_argument(
        '--validate-path', type=str, default='./validate', help='Path to validation data'
    )

    parser.add_argument(
        '--features', type=str, default='./models/features.pickle',
        help='Path to optional custom pre-computed feature values'
    )

    parser.add_argument(
        '--feature-model', type=str, default='./models/features.ckpt',
        help='Path to optional custom pre-trained feature model'
    )
    parser.add_argument(
        '--retrieval-model', type=str, default='./models/retrieval.ckpt',
        help='Path to optional custom pre-trained retrieval model'
    )

    parser.add_argument(
        '--training-data', type=str, default='./models/training-data.pickle',
        help='Path to optional pre-saved training data set'
    )

    parser.add_argument(
        '--learning-rate', type=float, default=0.01, help='Learning rate during training'
    )
    parser.add_argument(
        '--training-epochs', type=int, default=10, help='Number of epochs during training'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100, help='Batch size during training'
    )

    parser.add_argument(
        '--threshold', type=float, default=0.0, help='Threshold for cutting output'
    )

    parser.add_argument(
        '--train-feature-model', action='store_true', help='Train feature model'
    )
    parser.add_argument(
        '--generate-features', action='store_true', help='Generate feature values'
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Make sure we have generated a list of train IDS and their labels stored in a pickle
    train_labels = generate_dict_from_directory(args.train_path)

    if args.train_feature_model:
        train_feature_model(train_labels, args)

        sys.exit(0)

    if args.generate_features:
        generate_features([args.train_path, args.validate_path], args)

        sys.exit(0)

    # Optionally train the network
    if args.train:
        train(args)

    # Make sure we have generated a list of test IDS and their labels stored in a pickle
    test_labels = generate_dict_from_directory(args.test_path)

    all_labels = {}
    all_labels.update(train_labels)
    all_labels.update(test_labels)

    # Run the testing
    run_test(all_labels, list(test_labels.keys()), args)


if __name__ == "__main__":
    main()
