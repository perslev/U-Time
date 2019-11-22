import matplotlib.pyplot as plt
import numpy as np


def plot_period(X, y=None,
                channel_names=None,
                init_second=None,
                sample_rate=None,
                out_path=None,
                return_fig=False):
    """
    Plots one period (typically 30 seconds) of PSG data with its annotation.
    If neither out_path and return_fig are set, displays the figure and blocks.

    Args:
        X:                  (list)   A list of ndarrays of PSG periods
        y:                  (string) The epoch stage string (optional)
        channel_names:      (list)   A list of channel names, length equal to
                                     second dimension of 'X'.
        init_second         (int)    Show the time of this period in the axes
                                     title. If not set, no title will be set.
        sample_rate         (int)    The sample rate of the PSG ('X'). Used to
                                     compute the length of the period.
        out_path:           (string) Optional path to save the figure to
        return_fig:         (bool)   Return the figure instead of saving
                                     (out_path is ignored)

    Returns:
        Figure and axes objects if return_fig=True, otherwise None
    """
    X = X.squeeze()
    if X.ndim == 1:
        X = np.expand_dims(X, -1)
    n_chans = X.shape[-1]
    assert len(channel_names) == n_chans
    fig, axes = plt.subplots(figsize=(14, 7), ncols=1, nrows=n_chans,
                             sharex=True)
    fig.subplots_adjust(hspace=0)
    if n_chans == 1:
        axes = [axes]

    xs = np.arange(len(X))
    for i in range(n_chans):
        axes[i].plot(xs, X[:, i], color="black")
        axes[i].axhline(0, color='red',
                        linewidth=1.5)
        axes[i].set_xlim(xs[0], xs[-1])
        if channel_names:
            axes[i].annotate(
                s=channel_names[i],
                size=max(23-(2*len(channel_names)), 7),
                xy=(1.025, 0.5),
                xycoords=axes[i].transAxes,
                rotation=-90,
                va="center",
                ha="center"
            )
    p = "Period {}s".format(init_second) if init_second else ""
    p += "-{}s".format(init_second + int(len(X) / sample_rate)) if (init_second
                                                                    and sample_rate) else ""
    if p:
        axes[0].set_title(p, size=26)
    if isinstance(y, str):
        fig.suptitle("Sleep stage: {}".format(y), size=18)

    # Return, save or show the figure
    if not return_fig:
        if out_path:
            fig.savefig(out_path)
        else:
            plt.show()
        plt.close(fig)
    else:
        return fig, axes


def plot_periods(X, y=None,
                 highlight_periods=True,
                 out_path=None,
                 return_fig=False,
                 **kwargs):
    """
    Plots multiple consecutive periods of PSG data with annotated labels.
    If neither out_path and return_fig are set, displays the figure and blocks.

    Args:
        X:                  (list)   A list of ndarrays of PSG periods
        y:                  (list)   A list of epoch stage strings (optional)
        highlight_periods:  (bool)   Plot vertical lines to separate epochs
        out_path:           (string) Optional path to save the figure to
        return_fig:         (bool)   Return the figure instead of saving
                                     (out_path is ignored)
        **kwargs:           (dict)   Parameters passed to 'plot_period'

    Returns:
        Figure and axes objects if return_fig=True, otherwise None
    """
    X = np.array(X)
    if X.ndim == 3:
        X = np.concatenate(X, axis=0)
    if y is not None:
        if len(y) < 15:
            ys = '-'.join(y)
        else:
            ys = "{} stages (too long to show)".format(len(y))
    else:
        ys = "<Not specified>"
    fig, axes = plot_period(X, ys, return_fig=True, **kwargs)
    x_sepparations = [(len(X)//len(y)) * i for i in range(1, len(y))]
    if highlight_periods:
        for ax in axes:
            for sep in x_sepparations:
                ax.axvline(sep, color='red',
                           linestyle='--',
                           linewidth=1.5)

    # Return, save or show the figure
    if not return_fig:
        if out_path:
            fig.savefig(out_path)
        else:
            plt.show()
        plt.close(fig)
    else:
        return fig, axes
