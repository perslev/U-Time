"""
Script to compute confusion matrices from one or more pairs of true/pred .npz
files of labels
"""

from argparse import ArgumentParser
from glob import glob
import os
import numpy as np
import pandas as pd
from utime.evaluation import concatenate_true_pred_pairs
from sklearn.metrics import confusion_matrix
from utime.evaluation import (f1_scores_from_cm, precision_scores_from_cm,
                              recall_scores_from_cm)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Output a confusion matrix computed '
                                        'over one or more true/pred .npz '
                                        'files.')
    parser.add_argument("--true_pattern", type=str,
                        default="split*/predictions/test_data/files/*/true.npz",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--pred_pattern", type=str,
                        default="split*/predictions/test_data/files/*/pred.npz",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--normalized", action="store_true",
                        help="Normalize the CM to show fraction of total trues")
    parser.add_argument("--show_pairs", action="store_true",
                        help="Show the paired files (for debugging)")
    parser.add_argument("--round", type=int, default=3,
                        help="Round float numbers, only applicable "
                             "with --normalized.")
    parser.add_argument("--wake_trim_min", type=int, required=False,
                        help="Only evaluate on within wake_trim_min of wake "
                             "before and after sleep, as determined by true "
                             "labels")
    parser.add_argument("--period_length_sec", type=int, default=30,
                        help="Used with --wake_trim_min to determine number of"
                             " periods to trim")
    return parser


def wake_trim(pairs, wake_trim_min, period_length_sec):
    """
    Trim the pred/true pairs to remove long stretches of 'wake' in either end.
    Trims to a maximum of 'wake_trim_min' of uninterrupted 'wake' in either
    end, determined by the >TRUE< labels.

    args:
        pairs:            (list) A list of (true, prediction) pairs to trim
        wake_trim_min:    (int)  Maximum number of minutes of uninterrupted wake
                                 sleep stage (integer value '0') to allow
                                 according to TRUE values.
        period_length_sec (int)  The length in seconds of 1 period/epoch/segment

    Returns:
        List of trimmed (true, prediction) pairs
    """
    trim = int((60/period_length_sec) * wake_trim_min)
    trimmed_pairs = []
    for true, pred in pairs:
        inds = np.where(true != 0)[0]
        start = max(0, inds[0]-trim)
        end = inds[-1]+trim
        trimmed_pairs.append([
            true[start:end], pred[start:end]
        ])
    return trimmed_pairs


def trim(p1, p2):
    """
    Trims a pair of label arrays (true/pred normally) to equal length by
    removing elements from the tail of the longest array.
    This assumes that the arrays are aligned to the first element.
    """
    diff = len(p1) - len(p2)
    if diff > 0:
        p1 = p1[:len(p2)]
    else:
        p2 = p2[:len(p1)]
    return p1, p2


def run(args):
    """
    Run the script according to 'args' - Please refer to the argparser.
    """
    print("Looking for files...")
    k = lambda x: os.path.split(os.path.split(x)[0])[-1]
    true = sorted(glob(args.true_pattern), key=k)
    pred = sorted(glob(args.pred_pattern), key=k)
    if not true:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(args.true_pattern))
    if not pred:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(args.pred_pattern))
    if len(true) != len(pred):
        raise OSError("Did not find a matching number "
                      "of true and pred files ({} and {})"
                      "".format(len(true), len(pred)))
    if len(true) != len(set(true)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'true' files. Cannot uniquely match true/pred "
                         "files")
    if len(pred) != len(set(pred)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'pred' files. Cannot uniquely match true/pred "
                         "files")

    pairs = list(zip(true, pred))
    if args.show_pairs:
        print("PAIRS:\n{}".format(pairs))
    # Load the pairs
    print("Loading {} pairs...".format(len(pairs)))
    l = lambda x: [np.load(f)["arr_0"] for f in x]
    np_pairs = list(map(l, pairs))
    for i, (p1, p2) in enumerate(np_pairs):
        if len(p1) != len(p2):
            print("Not equal lengths: ", pairs[i], "{}/{}".format(len(p1),
                                                                  len(p2)))
            np_pairs[i] = trim(p1, p2)
    if args.wake_trim_min:
        print("OBS: Wake trimming of {} minutes (period length {} sec)"
              "".format(args.wake_trim_min, args.period_length_sec))
        np_pairs = wake_trim(np_pairs,
                             args.wake_trim_min,
                             args.period_length_sec)
    true, pred = concatenate_true_pred_pairs(pairs=np_pairs)
    cm = confusion_matrix(true, pred)
    if args.normalized:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)

    # Pretty print
    classes = len(cm)
    cm = pd.DataFrame(data=cm,
                      index=["True {}".format(i) for i in range(classes)],
                      columns=["Pred {}".format(i) for i in range(classes)])
    p = "Raw" if not args.normalized else "Normed"
    print(f"\n{p} Confusion Matrix:\n")
    print(cm.round(args.round))

    # Print metrics
    f1 = f1_scores_from_cm(cm)
    prec = precision_scores_from_cm(cm)
    recall = recall_scores_from_cm(cm)
    metrics = pd.DataFrame({
        "F1": f1,
        "Precision": prec,
        "Recall/Sens.": recall
    }, index=["Class {}".format(i) for i in range(classes)])
    metrics = metrics.T
    metrics["mean"] = metrics.mean(axis=1)
    print(f"\n{p} Metrics:\n")
    print(np.round(metrics.T, args.round), "\n")


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
