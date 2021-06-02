import os
from utime.preprocessing.dataset_preparation.utils import (download_and_validate,
                                                           get_checksums_and_file_names)

# Get path to current module file
_FILE_PATH = os.path.split(__file__)[0]

# SEDF-SC globals
_SERVER_URL_SC = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
_CHECKSUM_FILE_SC = "{}/sedf_sc_checksums.txt".format(_FILE_PATH)

# SEDF-ST globals
_SERVER_URL_ST = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry"
_CHECKSUM_FILE_ST = "{}/sedf_st_checksums.txt".format(_FILE_PATH)


def _download(out_dataset_folder, server_url, checksums_path, N_first=None):
    """ Download a sleep-EDF dataset """
    checksums, file_names = get_checksums_and_file_names(checksums_path)
    zipped = list(zip(file_names, checksums))
    N_first = int(N_first*2) if N_first is not None else len(file_names)
    zipped = zipped[:N_first]
    for i, (file_name, md5) in enumerate(zipped):
        print(f"Downloading {file_name} ({i + 1}/{len(zipped)})")
        out_subject_folder = file_name.split("-")[0][:-1] + "0"
        out_subject_folder = os.path.join(out_dataset_folder,
                                          out_subject_folder)
        if not os.path.exists(out_subject_folder):
            os.mkdir(out_subject_folder)
        out_file_path = os.path.join(out_subject_folder, file_name)
        download_url = server_url + "/{}".format(file_name)
        download_and_validate(download_url, md5, out_file_path)


def download_sedf_sc(out_dataset_folder, N_first=None):
    """ Download the Sleep-EDF sleep-cassette (153 records) dataset """
    _download(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL_SC,
        checksums_path=_CHECKSUM_FILE_SC,
        N_first=N_first
    )


def download_sedf_st(out_dataset_folder, N_first=None):
    """ Download the Sleep-EDF sleep-telemetry (44 records) dataset """
    _download(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL_ST,
        checksums_path=_CHECKSUM_FILE_ST,
        N_first=N_first
    )
