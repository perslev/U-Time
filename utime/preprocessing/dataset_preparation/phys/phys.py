import os
from glob import glob
from utime.preprocessing.dataset_preparation.utils import download_dataset

# Get path to current module file
_FILE_PATH = os.path.split(__file__)[0]

# Server base URL
_SERVER_URL = "https://physionet.org/files/challenge-2018/1.0.0"
_CHECKSUM_FILE = "{}/phys_checksums.txt".format(_FILE_PATH)


def phys_paths_func(file_name, server_url, out_dataset_folder):
    """
    See utime/preprocessing/dataset_preparation/utils.py [download_dataset]
    A callable of signature func(file_name, server_url, out_dataset_folder) which returns:
    1) download_url (path to fetch file from on remote system)
    2) out_file_path (path to store file on local system)
    """
    download_url = server_url + "/{}".format(file_name)
    out_subject_folder, file_name = file_name.replace("training/", "").split("/")
    out_file_path = os.path.join(out_dataset_folder, out_subject_folder, file_name)
    return download_url, out_file_path


def download_phys(out_dataset_folder, N_first=None):
    """ Download the DCSM (255 records) dataset """
    return download_dataset(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL,
        checksums_path=_CHECKSUM_FILE,
        paths_func=phys_paths_func,
        N_first=N_first*3 if N_first else None  # Three items per subject
    )


def preprocess_phys_hypnograms(dataset_folder_path):
    """
    Preprocesses files from the PHYS dataset.
    OBS: Only processes the hypnogram (.arousal) files
         Creates 1 new file in each PHYS subject dir (.ids format)

    :param dataset_folder_path: path to PHYS file on local disk
    :return: None
    """
    import numpy as np
    from wfdb.io import rdann
    from utime.io.high_level_file_loaders import load_psg
    from utime.bin.extract_hypno import to_ids
    from utime.hypnogram import SparseHypnogram
    from utime import Defaults

    # Get list of subject folders
    subject_folders = glob(os.path.join(dataset_folder_path, "tr*"))
    LABEL_MAP = {
        'N1': "N1",
        'N2': "N2",
        'N3': "N3",
        'R': "REM",
        'W': "W",
    }

    for i, folder in enumerate(subject_folders):
        name = os.path.split(os.path.abspath(folder))[-1]
        print(f"{i+1}/{len(subject_folders)}", name)

        # Get sleep-stages
        edf_file = folder + f"/{name}.mat"
        org_hyp_file = folder + f"/{name}.arousal"
        new_hyp_file = folder + f"/{name}.arousal.st"
        out_path = new_hyp_file.replace(".arousal.st", "-HYP.ids")
        if os.path.exists(out_path):
            print("Exists, skipping...")
            continue
        if os.path.exists(org_hyp_file):
            os.rename(org_hyp_file, new_hyp_file)

        psg, header = load_psg(edf_file, load_channels=['C3-M2'])
        hyp = rdann(new_hyp_file[:-3], "st")

        sample_rate = header["sample_rate"]
        psg_length_sec = len(psg)/sample_rate

        pairs = zip(hyp.aux_note, hyp.sample)
        stages = [s for s in pairs if not ("(" in s[0] or ")" in s[0])]
        stages = [(s[0], int(s[1]/sample_rate)) for s in stages]
        stages, starts = map(list, zip(*stages))

        if starts[0] != 0:
            i = [0] + starts
            s = ["UNKNOWN"] + [LABEL_MAP[s] for s in stages]
        else:
            i, s = starts, stages
        diff = psg_length_sec - i[-1]
        assert diff >= 0
        d = list(np.diff(i)) + [(diff//30) * 30]
        SparseHypnogram(i, d, [Defaults.get_stage_string_to_class_int()[s_] for s_ in s], 30)
        to_ids(i, d, s, out_path)
