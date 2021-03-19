import os
from utime.preprocessing.dataset_preparation.utils import (download_and_validate,
                                                           get_checksums_and_file_names)

# Get path to current module file
_FILE_PATH = os.path.split(__file__)[0]

# Server base URL
_SERVER_URL = "https://sid.erda.dk/share_redirect/fUH3xbOXv8/"
_CHECKSUM_FILE = "{}/dcsm_checksums.txt".format(_FILE_PATH)


def _download(out_dataset_folder, server_url, checksums_path, N_first=None):
    """ Download a sleep-EDF dataset """
    checksums, file_names = get_checksums_and_file_names(checksums_path)
    zipped = list(zip(file_names, checksums))
    N_first = int(N_first*2) if N_first is not None else len(file_names)
    zipped = zipped[:N_first]
    for i, (file_name, md5) in enumerate(zipped):
        print(f"Downloading {file_name} ({i + 1}/{len(zipped)})")
        download_url = server_url + "/{}".format(file_name)
        out_subject_folder, file_name = file_name.split("/")
        out_subject_folder = os.path.join(out_dataset_folder, out_subject_folder)
        out_file_path = os.path.join(out_subject_folder, file_name)
        if not os.path.exists(out_subject_folder):
            os.mkdir(out_subject_folder)
        download_and_validate(download_url, md5, out_file_path)


def download_dcsm(out_dataset_folder, N_first=None):
    """ Download the DCSM (255 records) dataset """
    return _download(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL,
        checksums_path=_CHECKSUM_FILE,
        N_first=N_first
    )
