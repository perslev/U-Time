import matplotlib.pyplot as plt
import numpy as np
from utime import Defaults


def get_reordered_hypnogram(hyp_array, annotation_dict, order):
    """
    Takes a ndarray-like hypnogram 'hyp_array' of integers and returns a re-leveled version
    according to 'order'. The 'order' should be specified as a list of sleep stage strings.
    The mapping between the integers in 'hyp_array' and the string representation in 'order' should be
    given by a dict 'annotation_dict' with integer key pointing to string sleep stages.

    E.g. an input array [0, 0, 1, 2, 3, 4] with order ['W', 'REM', 'N1', 'N2', 'N3'] and annotation_dict
    {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"} would produce the output:

    >> array([0, 0, 2, 3, 4, 1])
    """
    str_to_int_map = {value: key for key, value in annotation_dict.items()}
    int_order = [str_to_int_map[key] for key in order]
    map_ = {
        int(original_int): reordered_int for original_int, reordered_int in zip(int_order, range(len(int_order)))
    }
    mapped = []
    for stage in hyp_array:
        if int(stage) in map_:
            mapped.append(map_[int(stage)])
        else:
            mapped.append(np.nan)
    return np.asarray(mapped)


def plot_hypnogram(hyp_array,
                   true_hyp_array=None,
                   seconds_per_epoch=30,
                   annotation_dict=None,
                   show_f1_scores=True,
                   wake_trim_min=None,
                   order=("N3", "N2", "N1", "REM", "W")):
    """
    Plot a ndarray hypnogram of integers, 'hyp_array', optionally on top of an expert annotated hypnogram
    'true_hyp_array'.

    Args:
        hyp_array:         ndarray, shape [N]
        true_hyp_array:    ndarray, shape [N] (optional, default=None)
        seconds_per_epoch: integer, default=30
        annotation_dict:   dict, integer -> stage string mapping
        show_f1_scores:    bool, annotate the figure with f1 scores, only if true_hyp_array is set
        wake_trim_min:     None or integer, if set a number of max minutes before/proceeding the first/last
                           non-wake sleep stage in TRUE array to consider/'zoom' in on.
        order:             list-like of strings, order of sleep stages on plot y-axis

    Returns:
        fig, axes
    """
    rows = 1 if true_hyp_array is None else 2
    hight = 3 if true_hyp_array is None else 4
    fig, axes = plt.subplots(nrows=rows, figsize=(10, hight), sharex=True, sharey=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # Map classes to default string classes
    annotation_dict = annotation_dict or Defaults.get_class_int_to_stage_string()
    str_to_int_map = {value: key for key, value in annotation_dict.items()}

    # Define range in hours
    x_hours = np.array([seconds_per_epoch * i for i in range(len(hyp_array))]) / 3600

    if wake_trim_min and true_hyp_array is not None:
        trim = int((60/seconds_per_epoch) * wake_trim_min)
        inds = np.where(true_hyp_array != str_to_int_map["W"])[0]
        start = max(0, inds[0] - trim)
        end = inds[-1] + trim
        hyp_array = hyp_array[start:end]
        true_hyp_array = true_hyp_array[start:end]
        x_hours = x_hours[start:end]

    reordered_hyp_array = get_reordered_hypnogram(hyp_array, annotation_dict, order)
    axes[0].step(x_hours, reordered_hyp_array, where='post', color="black", label="Predicted")

    # Set ylabels
    for ax in axes:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    if true_hyp_array is not None:
        fig.text(x=0.02, y=0.56, s="Sleep Stage", ha="center", va="center", fontdict={"size": 16}, rotation=90)
    else:
        axes[0].set_ylabel("Sleep Stage", labelpad=12, size=16)
    axes[-1].set_xlabel("Time (hours)", size=16, labelpad=12)
    axes[-1].set_xlim(x_hours[0], x_hours[-1])

    fig.tight_layout()
    if true_hyp_array is not None:
        reordered_true = get_reordered_hypnogram(true_hyp_array, annotation_dict, order)
        axes[1].step(x_hours, reordered_true, where='post', color="darkred", label="Expert")

        if show_f1_scores:
            from sklearn.metrics import f1_score
            labels = [str_to_int_map[w] for w in reversed(order)]
            mask = np.isin(true_hyp_array.flatten(), np.array(labels).flatten())
            f1s = f1_score(true_hyp_array[mask], hyp_array[mask], labels=labels, average=None, zero_division=1)
            f1s = [round(l, 2) for l in (list(f1s) + [np.mean(f1s)])]
            f1_labels = list(reversed(order)) + ["Mean"]

            # Plot title
            fig.text(
                x=0.845,
                y=0.66,
                s="F1-scores",
                ha="left",
                va="top",
                fontdict={"alpha": 1, "size": 14}
            )
            for i, (stage, value) in enumerate(zip(f1_labels, f1s)):
                # Plot stage
                fig.text(
                    x=0.845,
                    y=0.58-(0.05*i),
                    s=f"{stage}",
                    ha="left",
                    va="top",
                    fontdict={"alpha": 1, "size": 14}
                )
                # Plot score
                fig.text(
                    x=0.9475,
                    y=0.58-(0.05*i),
                    s="{:.2f}".format(value),
                    ha="left",
                    va="top",
                    fontdict={"alpha": 1, "size": 14}
                )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(l, []) for l in zip(*lines_labels)]
        l = fig.legend(lines, labels, loc='right', bbox_to_anchor=(1.01, 0.89), ncol=1, fontsize=14)
        l.get_frame().set_linewidth(0)
        fig.subplots_adjust(hspace=0.13, right=0.81, left=0.10)
    return fig, axes
