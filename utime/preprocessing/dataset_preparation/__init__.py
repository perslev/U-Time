"""
Support for additional datasets will be added over time
"""

from .sedf import download_sedf_sc, download_sedf_st


DOWNLOAD_FUNCS = {
    "sedf_sc": download_sedf_sc,
    "sedf_st": download_sedf_st
}


def no_processing(*args, **kwargs):
    pass


PREPROCESS_FUNCS = {
    "sedf_sc": no_processing,
    "sedf_st": no_processing
}


def download_dataset(dataset_name, out_dir, N_first=None):
    DOWNLOAD_FUNCS[dataset_name](out_dir, N_first)


def preprocess_dataset(dataset_name, out_dir):
    PREPROCESS_FUNCS[dataset_name](out_dir)
