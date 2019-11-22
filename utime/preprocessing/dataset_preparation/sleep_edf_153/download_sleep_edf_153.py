import hashlib
import subprocess
import os

_FILE_PATH = os.path.split(__file__)[0]
_SERVER_URL = "https://physionet.org/physiobank/database/" \
              "sleep-edfx/sleep-cassette"


def get_checksums_and_file_names():
    """ Reads the local checksums.txt file """
    with open("{}/checksums.txt".format(_FILE_PATH)) as in_f:
        return zip(*[l.strip(" \n\r").split("  ") for l in in_f.readlines()])


def validate_md5(local_path, md5):
    """
    Computes the md5 checksum of a file and compares it to the passed md5
    checksum
    """
    with open(local_path, "rb") as in_f:
        md5_file = hashlib.md5(in_f.read()).hexdigest()
        if md5_file != md5:
            return False
        else:
            return True


def download_and_validate(file_name, md5, out_path):
    """
    Download file 'file_name' and validate md5 checksum against 'md5'. Saves
    the downloaded file to 'out_path'.
    If the file already exists, and have a valid md5, the download is skipped.
    """
    if os.path.exists(out_path) and validate_md5(out_path, md5):
        print("... skipping (already downloaded with valid md5)")
        return

    download_url = _SERVER_URL + "/{}".format(file_name)
    p = subprocess.Popen(["wget", download_url, "-O", out_path],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()

    if not validate_md5(out_path, md5):
        os.remove(out_path)
        raise ValueError(f"Invalid md5 for file {file_name} "
                         f"(please restart download)")


def download_sleep_edf_153(out_dataset_folder, N_first=None):
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
        download_and_validate(file_name, md5, out_file_path)
