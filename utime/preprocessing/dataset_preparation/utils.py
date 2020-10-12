import os
import hashlib
import subprocess


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


def download_and_validate(download_url, md5, out_path):
    """
    Download file 'file_name' and validate md5 checksum against 'md5'. Saves
    the downloaded file to 'out_path'.
    If the file already exists, and have a valid md5, the download is skipped.
    """
    if os.path.exists(out_path) and validate_md5(out_path, md5):
        print("... skipping (already downloaded with valid md5)")
        return
    p = subprocess.Popen(["wget", download_url, "-O", out_path],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    p.communicate()
    if not validate_md5(out_path, md5):
        os.remove(out_path)
        raise ValueError(f"Invalid md5 for file at {download_url} "
                         f"(please restart download)")
