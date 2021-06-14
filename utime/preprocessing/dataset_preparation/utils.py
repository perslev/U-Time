import os
import hashlib
import requests
from pathlib import Path


def get_checksums_and_file_names(path):
    """ Reads the local checksums file """
    with open(path) as in_f:
        return zip(*[map(lambda x: x.strip('\n\r\t '), l.strip(" ").split(" ", maxsplit=1)) for l in in_f.readlines()])


def get_sha256(local_path):
    """
    Computes the sha256 of a local file.

    :param local_path: string/path to file
    :return: sha256 checksum
    """
    file_hash = hashlib.sha256()
    with open(local_path, "rb") as in_f:
        for chunk in iter(lambda: in_f.read(512 * file_hash.block_size), b''):
            file_hash.update(chunk)
    return file_hash


def validate_sha256(local_path, sha256):
    """
    Computes the sha256 checksum of a file and compares it to the passed sha256 hexdigest checksum
    """
    file_hash = get_sha256(local_path)
    return file_hash.hexdigest() == sha256


def download_and_validate(download_url, sha256, out_path):
    """
    Download file 'file_name' and validate sha256 checksum against 'sha256'.
    Saves the downloaded file to 'out_path'.
    If the file already exists, and have a valid sha256, the download is skipped.
    """
    if os.path.exists(out_path):
        if validate_sha256(out_path, sha256):
            print("... skipping (already downloaded with valid sha256)")
            return
        else:
            print("... File exists, but invalid SHA256, re-downloading")

    response = requests.get(download_url, allow_redirects=True)
    if response.ok:
        with open(out_path, "wb") as out_f:
            out_f.write(response.content)
    else:
        raise ValueError("Could not download file from URL {}. "
                         "Received HTTP response with status code {}".format(download_url,
                                                                             response.status_code))
    if not validate_sha256(out_path, sha256):
        os.remove(out_path)
        raise ValueError(f"Invalid sha256 for file at {download_url} "
                         f"(please restart download)")


def get_n_first(file_names, checksums, N_first):
    """
    Given a list of file_names and a list of checksums, returns the 'N_first'
    items from both lists as a zip object.

    :param file_names: list of file names
    :param checksums: list of sha256 checksums
    :param N_first: int or None. If None, all items are returned
    :return: zipped N_first first items of file_names and checksums
    """
    zipped = list(zip(file_names, checksums))
    N_first = int(N_first) if N_first is not None else len(file_names)
    zipped = zipped[:N_first]
    return zipped


def download_dataset(out_dataset_folder, server_url, checksums_path, paths_func, N_first=None):
    """
    Download a dataset into 'out_dataset_folder' by fetching files at URL 'server_url' according to
    the list of checksums and filenames in file 'checksums_path'. Only downloads the N_first subject folders
    if 'N_first' is specified.

    'paths_func' should be a callable of signature func(file_name, server_url, out_dataset_folder) which returns:
        1) download_url (path to fetch file from on remote system)
        2) out_file_path (path to store file on local system)
    """
    checksums, file_names = get_checksums_and_file_names(checksums_path)
    zipped = get_n_first(file_names, checksums, N_first)
    for i, (file_name, sha256) in enumerate(zipped):
        print(f"Downloading {file_name} ({i + 1}/{len(zipped)})")
        download_url, out_file_path = paths_func(file_name, server_url, out_dataset_folder)
        out_file_path = Path(out_file_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        download_and_validate(download_url, sha256, out_file_path)
