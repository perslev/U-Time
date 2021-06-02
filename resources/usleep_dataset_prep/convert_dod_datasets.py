import os
import h5py
from glob import glob
import json
import numpy as np
from utime.utils.scriptutils.extract import to_h5_file


def convert_h5_file(h5_path):
    out_dir = h5_path.replace(".h5", "")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_h5_path = os.path.join(out_dir, "signals.h5")
    out_hyp_path = os.path.join(out_dir, "hypnogram.npy")
    if os.path.exists(out_h5_path) and os.path.exists(out_hyp_path):
        return

    with h5py.File(h5_path, "r") as in_f:
        description = json.loads(in_f.attrs['description'])

        data = []
        sample_rates = []
        channel_names = []
        for channel in description:
            sample_rates.append(int(channel['fs']))
            data.append(np.array(in_f[channel["path"]]))
            channel_names.append(channel["path"].split("/")[-1].replace("_", "-"))

        assert np.all(np.array(sample_rates) == sample_rates[0])
        sample_rate = sample_rates[0]
        data = np.array(data).T

        print(channel_names)
        to_h5_file(
            out_path=out_h5_path,
            data=data,
            sample_rate=sample_rate,
            channel_names=channel_names,
            date=None
        )

        # Save hypnogram
        hyp = np.array(in_f['hypnogram'])
        np.save(out_hyp_path, hyp)


if __name__ == "__main__":
    files = glob("dod_*/*h5")

    for file_ in files:
        print(file_)
        convert_h5_file(file_)
