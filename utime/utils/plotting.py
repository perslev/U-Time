import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

logger = logging.getLogger(__name__)


def plot_all_training_curves(glob_path, out_path, **kwargs):
    paths = glob(glob_path)
    if not paths:
        raise OSError(f"File pattern {glob_path} gave no matches matches '({paths})'")
    out_folder = os.path.split(out_path)[0]
    for p in paths:
        if len(paths) > 1:
            # Set unique names
            uniq = os.path.splitext(os.path.split(p)[-1])[0]
            f_name = uniq + "_" + os.path.split(out_path)[-1]
            save_path = os.path.join(out_folder, f_name)
        else:
            save_path = out_path
        plot_training_curves(p, save_path, **kwargs)


def plot_training_curves(csv_path, save_path, logy=False,
                         exclude=("learning_rate", "epoch", "loss",
                                  "epoch_minutes", "train_hours",
                                  'memory_usage_gib'),
                         include_regex=None):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Prepare plot
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(311)

    # Get epoch, training and validation loss vectors
    epochs = df["epoch"] + 1
    train_loss = df["loss"]
    val_loss = df.get("val_loss")

    if logy:
        train_loss = np.log10(train_loss)
        if val_loss is not None:
            val_loss = np.log10(val_loss)

    # Plot
    ax1.plot(epochs, train_loss, lw=3, color="darkblue", label="Training loss")
    if val_loss is not None:
        ax1.plot(epochs, val_loss, lw=3, color="darkred", label="Validation loss")

    # Add legend, labels and title
    leg = ax1.legend(loc=0)
    leg.get_frame().set_linewidth(0)
    ax1.set_xlabel("Epoch", size=16)
    ax1.set_ylabel("Loss" if not logy else "$\log_{10}$(Loss)", size=16)
    ax1.set_title("Training %sloss" % ("and validation " if val_loss is not None else ""), size=20)

    # Make second plot
    ax2 = fig.add_subplot(312)

    # Get all other columns, optionally only if matching 'include_regex'
    import re
    include_regex = re.compile(include_regex or ".*")

    plotted = 0
    for col in df.columns:
        if any([s in col for s in exclude[1:]]) or col == "lr":
            continue
        elif not re.match(include_regex, col):
            continue
        else:
            plotted += 1
            ax2.plot(epochs, df[col], label=col, lw=2)

    # Add legend, labels and title
    if plotted <= 8:
        # Otherwise it takes up all the space
        leg = ax2.legend(loc=0, ncol=int(np.ceil(plotted/5)))
        leg.get_frame().set_linewidth(0)
    ax2.set_xlabel("Epoch", size=16)
    ax2.set_ylabel("Metric", size=16)
    ax2.set_title("Training and validation metrics", size=20)

    # Plot learning rate
    lr = df.get("lr")
    if lr is None:
        lr = df.get("learning_rate")
    if lr is not None:
        ax3 = fig.add_subplot(313)
        ax3.step(epochs, lr)
        ax3.set_xlabel("Epoch", size=16)
        ax3.set_ylabel("Learning Rate", size=16)
        ax3.set_title("Learning Rate", size=20)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig.number)
