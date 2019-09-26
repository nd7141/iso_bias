import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Kernels')

    parser.add_argument(
        '--dataset', type=str,
        default='MUTAG',
        help='Dataset to train on')

    parser.add_argument(
        '--clean_dataset', default=False, action='store_true', dest='clean_dataset',
        help='Clean dataset')

    parser.add_argument(
        '--orbits_path', type=str,
        default='./orbits/no_labels/orbits/',
        help='Path to orbits')

    parser.add_argument(
        '--dir', type=str,
        default='./Datasets',
        help='Directory to save datasets to')

    parser.add_argument(
        '--kernel', type=str,
        default='WL',
        help='An abbreviated kernel name')

    parser.add_argument(
        '--parameter', type=str,
        default='5',
        help='Parameter(s) in a kernel')

    args = parser.parse_args()

    return args
