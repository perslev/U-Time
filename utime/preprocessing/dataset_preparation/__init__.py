"""
Support for additional datasets will be added over time
"""

from .sleep_edf_153 import download_sleep_edf_153


DOWNLOAD_FUNCS = {
    "sleep-EDF-153": download_sleep_edf_153
}


def no_processing(*args, **kwargs):
    pass


PREPROCESS_FUNCS = {
    "sleep-EDF-153": no_processing
}


def download_dataset(dataset_name, out_dir, N_first=None):
    DOWNLOAD_FUNCS[dataset_name](out_dir, N_first)


def preprocess_dataset(dataset_name, out_dir):
    PREPROCESS_FUNCS[dataset_name](out_dir)
