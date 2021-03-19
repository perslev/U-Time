import numpy as np
import os
from argparse import ArgumentParser
from scipy import stats
from glob import glob


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Majority vote across a set of channels.')
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help='Path to folder storing predictions for each dataset. '
                             'The specified folder must store sub-folders for each given dataset. '
                             'Each dataset folder must store results from each channel combination each '
                             'in a sub-folder named according to the channel combination.')
    parser.add_argument("--soft", action="store_true",
                        help="If using NxC shaped probability-like arrays, use this option to sum arrays "
                             "instead of computing mode.")
    return parser


def get_datasets(folder):
    """
    Returns a dictionary of dataset-ID: dataset directory paths
    """
    paths = glob(f"{folder}/*")
    return {os.path.split(p)[-1]: p for p in paths if os.path.isdir(p)}


def get_true_paths(dataset_dir):
    """
    Returns a dictionary of study-ID: true/target vector paths
    """
    true_paths = glob(f'{dataset_dir}/*TRUE.np*')
    return {
        os.path.split(p)[-1].split("_TRUE")[0]: p for p in true_paths
    }


def get_prediction_paths(pred_dir):
    """
    Returns a dictionary of study-ID: predicted vector paths
    """
    pred_paths = glob(f'{pred_dir}/*PRED.np*')
    return {
        os.path.split(p)[-1].split("_PRED")[0]: p for p in pred_paths
    }


def get_input_channel_combinations(dataset_dir, study_id):
    """
    Returns a dictionary of channel-combination string IDs: dir paths
    """
    # Find all directories in the dataset directory
    return glob(f'{dataset_dir}/*/*{study_id}_PRED.np*')


def get_arrays(paths):
    """
    Takes a dictionary of study-ID: numpy npy/npz file path and returns
    a dictionary of study-ID: loaded numpy arrays
    """
    loaded = []
    for arr_path in paths:
        loaded.append(np.load(arr_path))
    return np.stack(loaded)


def run(folder, soft):
    dataset_dirs = get_datasets(folder=folder)

    for dataset, dataset_dir_path in dataset_dirs.items():
        print(f"Processing dataset '{dataset}'")

        # Create majority vote folder
        out_dir = f'{dataset_dir_path}/majority'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Get all study IDs
        study_ids = set([os.path.split(s)[-1].split("_PRED")[0] for s in glob(dataset_dir_path + "/**/*PRED.npy")])
        print(f"Found {len(study_ids)} paths to study IDs")

        for study_id in study_ids:
            print(dataset, study_id)
            channels = get_input_channel_combinations(dataset_dir_path, study_id)
            channel_arrs = get_arrays(channels)

            # Compute MJ vote
            if soft:
                mj = np.mean(channel_arrs, axis=0).squeeze()
            else:
                mj = stats.mode(channel_arrs, axis=0)[0].squeeze()
            np.save(f'{out_dir}/{study_id}_PRED', mj)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    run(folder=args.dataset_dir, soft=args.soft)


if __name__ == "__main__":
    entry_func()
