import logging
import os
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def get_hypnogram(y_pred, y_true=None, id_=None):
    def format_ax(ax):
        ax.set_xlabel("Period number")
        ax.set_ylabel("Sleep stage")
        ax.set_yticks(range(6))
        ax.set_yticklabels(["Wake", "N1", "N2", "N3", "REM", "Unknown"])
        ax.invert_yaxis()
        ax.set_xlim(1, ids[-1]+1)
        l = ax.legend(loc=3)
        l.get_frame().set_linewidth(0)
    ids = np.arange(len(y_pred))
    if y_true is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    fig.suptitle("Hypnogram for identifier: {}".format(id_ or "???"))

    # Plot predicted hypnogram
    ax1.step(ids+1, y_pred, color="darkcyan", label="Predicted")
    format_ax(ax1)
    if y_true is not None:
        ax2.step(ids+1, y_true, color="darkred", label="True")
        format_ax(ax2)
        # fig.subplots_adjust(hspace=0.4)
        return fig, ax1, ax2
    return fig, ax1


def plot_and_save_hypnogram(out_path, y_pred, y_true=None, id_=None):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    outs = get_hypnogram(y_pred, y_true, id_)
    outs[0].savefig(out_path, dpi=180)
    plt.close(outs[0])


def plot_confusion_matrix(y_true, y_pred, n_classes,
                          normalize=False, id_=None,
                          cmap="Blues"):
    """
    Adapted from sklearn 'plot_confusion_matrix.py'.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    if normalize:
        title = 'Normalized confusion matrix for identifier {}'.format(id_ or "???")
    else:
        title = 'Confusion matrix, without normalization for identifier {}' \
                ''.format(id_ or "???")

    # Compute confusion matrix
    classes = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get transformed labels
    from utime import Defaults
    labels = [Defaults.get_class_int_to_stage_string()[i] for i in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def plot_and_save_cm(out_path, pred, true, n_classes, id_=None, normalized=True):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig, ax = plot_confusion_matrix(true, pred, n_classes, normalized, id_)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
