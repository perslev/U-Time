from argparse import ArgumentParser
from glob import glob
import os


def get_argparser():
    parser = ArgumentParser(description='Extract hypnograms from various'
                                        ' file formats.')
    parser.add_argument("--file_regex", type=str,
                        help='A glob statement matching all files to extract '
                             'from')
    parser.add_argument("--out_dir", type=str,
                        help="Directory in which extracted files will be "
                             "stored")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    return parser


def to_ids(start, durs, stage, out):
    with open(out, "w") as out_f:
        for i, d, s in zip(start, durs, stage):
            out_f.write("{},{},{}\n".format(i, d, s))


def load_xml(file_):
    import xml.etree.ElementTree as ET
    events = ET.parse(file_).findall('ScoredEvents')
    assert len(events) == 1
    stage_dict = {
        "Wake|0": "W",
        "Stage 1 sleep|1": "N1",
        "Stage 2 sleep|2": "N2",
        "Stage 3 sleep|3": "N3",
        "Stage 4 sleep|4": "N3",
        "REM sleep|5": "REM",
        "Movement|6": "UNKNOWN",
        "Unscored|9": "UNKNOWN"
    }
    starts, durs, stages = [], [], []
    for event in events[0]:
        if not event[0].text == "Stages|Stages":
            continue
        stage = stage_dict[event[1].text]
        start = int(float(event[2].text))
        dur = int(float(event[3].text))
        starts.append(start)
        durs.append(dur)
        stages.append(stage)
    return starts, durs, stages


EXT_TO_FUNC = {
    "xml": load_xml
}


def run(file_regex, out_dir, overwrite):
    files = glob(file_regex)
    out_dir = os.path.abspath(out_dir)
    n_files = len(files)
    print("Found {} files matching glob statement".format(n_files))
    if n_files == 0:
        return
    print("Saving .ids files to '{}'".format(out_dir))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        name = os.path.splitext(os.path.split(file_)[-1])[0]
        out = os.path.join(out_dir, name + ".ids")
        if os.path.exists(out):
            if not overwrite:
                continue
            os.remove(out)
        print("  {}/{} Processing {}".format(i+1, n_files, name),
              flush=True, end="\r")
        func = EXT_TO_FUNC[os.path.splitext(file_)[-1][1:]]
        start, dur, stage = func(file_)
        to_ids(start, dur, stage, out)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = vars(parser.parse_args(args))
    run(**args)


if __name__ == "__main__":
    entry_func()
