"""
Support for additional datasets will be added over time
"""

from .sedf import download_sedf_sc, download_sedf_st
from .dcsm import download_dcsm
from .phys import download_phys, preprocess_phys_hypnograms


DOWNLOAD_FUNCS = {
    "sedf_sc": download_sedf_sc,
    "sedf_st": download_sedf_st,
    "dcsm": download_dcsm,
    "phys": download_phys
}


def no_processing(*args, **kwargs):
    pass


PREPROCESS_FUNCS = {
    "phys": preprocess_phys_hypnograms
}


def download_dataset(dataset_name, out_dir, N_first=None):
    DOWNLOAD_FUNCS[dataset_name](out_dir, N_first=N_first)


def preprocess_dataset(dataset_name, out_dir):
    func = PREPROCESS_FUNCS.get(dataset_name, no_processing)
    print("Preprocessing folder '{}' with function '{}'".format(out_dir, func.__name__))
    func(out_dir)
