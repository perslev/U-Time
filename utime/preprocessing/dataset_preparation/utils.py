import os
import hashlib
import subprocess


def validate_sha256(local_path, sha256):
    """
    Computes the md5 checksum of a file and compares it to the passed md5
    checksum
    """
    with open(local_path, "rb") as in_f:
        sha256_file = hashlib.sha256(in_f.read()).hexdigest()
        if sha256_file != sha256:
            return False
        else:
            return True


def download_and_validate(download_url, sha256, out_path):
    """
    Download file 'file_name' and validate sha256 checksum against 'sha256'.
    Saves the downloaded file to 'out_path'.
    If the file already exists, and have a valid sha256, the download is skipped.
    """
    if os.path.exists(out_path) and validate_sha256(out_path, sha256):
        print("... skipping (already downloaded with valid sha256)")
        return
    p = subprocess.Popen(["wget", download_url, "-O", out_path],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()
    if not validate_sha256(out_path, sha256):
        os.remove(out_path)
        raise ValueError(f"Invalid sha256 for file at {download_url} "
                         f"(please restart download)")
