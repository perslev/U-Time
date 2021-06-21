import os
from utime.preprocessing.dataset_preparation.utils import download_dataset

# Get path to current module file
_FILE_PATH = os.path.split(__file__)[0]

# Server base URL
_SERVER_URL = "https://sid.erda.dk/share_redirect/fUH3xbOXv8/"
_CHECKSUM_FILE = "{}/dcsm_checksums.txt".format(_FILE_PATH)


def dcsm_paths_func(file_name, server_url, out_dataset_folder):
    """
    See utime/preprocessing/dataset_preparation/utils.py [download_dataset]
    A callable of signature func(file_name, server_url, out_dataset_folder) which returns:
    1) download_url (path to fetch file from on remote system)
    2) out_file_path (path to store file on local system)
    """
    download_url = server_url + "/{}".format(file_name)
    out_subject_folder, file_name = file_name.split("/")
    out_file_path = os.path.join(out_dataset_folder, out_subject_folder, file_name)
    return download_url, out_file_path


def download_dcsm(out_dataset_folder, N_first=None):
    """ Download the DCSM (255 records) dataset """
    return download_dataset(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL,
        checksums_path=_CHECKSUM_FILE,
        paths_func=dcsm_paths_func,
        N_first=N_first*2 if N_first else None  # Two items per subject
    )
