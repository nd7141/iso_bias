from constants import BIO, SOCIAL
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Graph')

    parser.add_argument(
        '--dataset', type=str,
        default='MUTAG',
        help='Dataset to train on')

    parser.add_argument(
        '--clean_dataset', default=False, action='store_true', dest='clean_dataset',
        help='Clean dataset')

    parser.add_argument(
        '--orbits_path', type=str,
        default='../results_no_labels/orbits/',
        help='Path to orbits')

    parser.add_argument(
        '--dir', type=str,
        default='./Datasets',
        help='Directory to save datasets to')

    parser.add_argument(
        '--num_epochs', type=int,
        default=350,
        help='Number of epochs to train for')

    parser.add_argument(
        '--num_kfold', type=int,
        default=10,
        help='Number of k folds')

    parser.add_argument(
        '--batch_size', type=int,
        default=32,
        help='Batch size')

    parser.add_argument(
        '-lr', type=int,
        default=0.01,
        help='Learning rate')

    parser.add_argument(
        '--randomize',
        type=int,
        default=None,
        help='If randomize node features (size of vector)')

    parser.add_argument(
        '--initialize_node_features',
        type=int,
        default=None,
        help='Whether to force initialization of ')

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout ratio')

    parser.add_argument(
        '--hidden',
        type=int,
        default=32,
        help='Hidden size')

    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train share')


    args = parser.parse_args()

    # assert args.dataset in BIO+SOCIAL, \
    #     "This dataset is not currently supported or doesn't exist."

    return args
