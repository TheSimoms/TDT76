import sys
import argparse
import logging

from feature_extractor import generate_features, train_feature_model
from utils import generate_dict_from_directory
from your_code import train, test


def main():
    """
    Run the system. Using run-time arguments, one are able to invoke different procedures;
    - Train feature model
    - Generate training data features
    - Train the network
    - Run the tester
    """

    # Parse run-time arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Debug
    parser.add_argument('--debug', action='store_true', help='Toggle debug')

    # Run-time changes
    parser.add_argument('--train', action='store_true', help='Train the network')
    parser.add_argument(
        '--threshold', type=float, default=0.65, help='Threshold for cutting retrieval output'
    )
    parser.add_argument(
        '--k', type=int, default=1000, help='Number of images to run tests on'
    )
    parser.add_argument(
        '--test-image', type=str, help='Run the test on a single image', metavar='Image ID'
    )

    # Paths to image folders
    parser.add_argument('--train-path', type=str, default='./train', help='Path to training data')
    parser.add_argument('--test-path', type=str, default='./test', help='Path to testing data')

    # Paths to pre-trained data
    parser.add_argument(
        '--features', type=str, default='./features/features.pickle',
        help='Path to pre-computed feature values'
    )
    parser.add_argument(
        '--feature-model', type=str, default='./models/features.ckpt',
        help='Path to pre-trained feature model'
    )
    parser.add_argument(
        '--retrieval-model', type=str, default='./models/retrieval.ckpt',
        help='Path to pre-trained retrieval model'
    )

    # Training parameters
    parser.add_argument(
        '--learning-rate', type=float, default=0.01, help='Learning rate during training'
    )
    parser.add_argument(
        '--training-epochs', type=int, default=1, help='Number of epochs during training'
    )
    parser.add_argument(
        '--number-of-batches', type=int, default=5000, help='Number of batches during training'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10, help='Batch size during training'
    )

    rebuild_system_warning = 'The system must be re-built when changing this parameter'

    # System parameters. When changing these, the system must be re-built
    parser.add_argument(
        '--label-threshold', type=float, default=0.01,
        help='How big percentage of images a label is used by for it to activate. This is done \
             for the model to fit into memory %s' % rebuild_system_warning
    )
    parser.add_argument(
        '--image-size', type=int, default=160,
        help='Resize images to quadrats of this size %s' % rebuild_system_warning
    )
    parser.add_argument(
        '--number-of-channels', type=int, default=3,
        help='Number of channels in images %s. 3 for RGB' % rebuild_system_warning
    )

    # Options for re-building the system
    parser.add_argument(
        '--train-feature-model', action='store_true', help='Train feature model'
    )
    parser.add_argument(
        '--generate-features', action='store_true', help='Generate feature values'
    )

    args = parser.parse_args()

    # Toggle debug if wanted
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Fetch training labels
    train_labels = generate_dict_from_directory(args.train_path)

    # Train feature model if wanted
    if args.train_feature_model:
        train_feature_model(train_labels, args)

        sys.exit(0)

    # Generate features if wanted
    if args.generate_features:
        generate_features(args)

        sys.exit(0)

    # Train the model if wanted
    if args.train:
        train(args)

        sys.exit(0)

    # Fetch testing labels
    test_labels = generate_dict_from_directory(args.test_path)

    # Collect all labels
    all_labels = {}
    all_labels.update(train_labels)
    all_labels.update(test_labels)

    # Run the testing
    test(all_labels, list(test_labels.keys()), args)


if __name__ == "__main__":
    main()
