import os
from utime.preprocessing.dataset_preparation.utils import download_and_validate

_FILE_PATH = os.path.split(__file__)[0]
_SERVER_URL = "https://physionet.org/physiobank/database/" \
              "sleep-edfx/sleep-cassette"


def get_checksums_and_file_names():
    """ Reads the local checksums.txt file """
    with open("{}/checksums.txt".format(_FILE_PATH)) as in_f:
        return zip(*[l.strip(" \n\r").split("  ") for l in in_f.readlines()])


def download_sedf_sc(out_dataset_folder, N_first=None):
    """ Download the sleep-EDF-153 dataset """
    checksums, file_names = get_checksums_and_file_names()
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
        download_url = _SERVER_URL + "/{}".format(file_name)
        download_and_validate(download_url, md5, out_file_path)
