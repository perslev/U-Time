"""
Support for additional datasets will be added over time
"""

from .sedf_sc import download_sedf_sc


DOWNLOAD_FUNCS = {
    "sedf_sc": download_sedf_sc
}


def no_processing(*args, **kwargs):
    pass


PREPROCESS_FUNCS = {
    "sedf_sc": no_processing
}


def download_dataset(dataset_name, out_dir, N_first=None):
    DOWNLOAD_FUNCS[dataset_name](out_dir, N_first)


def preprocess_dataset(dataset_name, out_dir):
    PREPROCESS_FUNCS[dataset_name](out_dir)
