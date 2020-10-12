"""
Support for additional datasets will be added over time
"""

from .sedf import download_sedf_sc, download_sedf_st
from .dcsm import download_dcsm


DOWNLOAD_FUNCS = {
    "sedf_sc": download_sedf_sc,
    "sedf_st": download_sedf_st,
    "dcsm": download_dcsm
}


def no_processing(*args, **kwargs):
    pass


PREPROCESS_FUNCS = {
}


def download_dataset(dataset_name, out_dir, N_first=None):
    DOWNLOAD_FUNCS[dataset_name](out_dir, N_first=N_first)


def preprocess_dataset(dataset_name, out_dir):
    PREPROCESS_FUNCS.get(dataset_name, no_processing)(out_dir)
