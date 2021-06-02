import os
import hashlib
import requests


def get_checksums_and_file_names(path):
    """ Reads the local checksums file """
    with open(path) as in_f:
        return zip(*[map(lambda x: x.strip('\n\r\t '), l.strip(" ").split(" ", maxsplit=1)) for l in in_f.readlines()])


def validate_sha256(local_path, sha256):
    """
    Computes the md5 checksum of a file and compares it to the passed sha256 hexdigest checksum
    """
    file_hash = hashlib.sha256()
    with open(local_path, "rb") as in_f:
        for chunk in iter(lambda: in_f.read(512 * file_hash.block_size), b''):
            file_hash.update(chunk)
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
