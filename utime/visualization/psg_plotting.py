import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from utime import Defaults
from utime.io.channels import ChannelMontageTuple


def set_equal_ylims(axes):
    mins, maxs = list(zip(*[ax.get_ylim() for ax in axes]))
    min_, max_ = np.min(mins), np.max(maxs)
    for ax in axes:
        ax.set_ylim(min_, max_)


def plot_period(X, y=None,
                channel_names=None,
                init_second=None,
                sample_rate=None,
                horizontal_line=True,
                out_path=None,
                return_fig=False,
                equal_y_lims=True,
                **plot_kwargs):
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
        equal_y_lims        (bool)   All axes share ylims

    Returns:
        Figure and axes objects if return_fig=True, otherwise None
    """
    X = X.squeeze()
    if X.ndim == 1:
        X = np.expand_dims(X, -1)
    n_chans = X.shape[-1]
    fig, axes = plt.subplots(figsize=(14, 7), ncols=1, nrows=n_chans,
                             sharex=True)
    fig.subplots_adjust(hspace=0)
    if n_chans == 1:
        axes = [axes]

    xs = np.arange(len(X))
    for i in range(n_chans):
        axes[i].plot(xs, X[:, i], color="black", **plot_kwargs)
        if horizontal_line:
            axes[i].axhline(0, color='red',
                            linewidth=1.5)
        axes[i].set_xlim(xs[0], xs[-1])
        if channel_names:
            assert len(channel_names) == n_chans
            if isinstance(channel_names, ChannelMontageTuple):
                channel_names = channel_names.original_names
            axes[i].annotate(
                text=channel_names[i],
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
    if equal_y_lims:
        set_equal_ylims(axes)

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
                 channel_names=None,
                 highlight_periods=True,
                 out_path=None,
                 return_fig=False,
                 equal_y_lims=True,
                 sample_rate=None,
                 **plot_kwargs):
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
        equal_y_lims        (bool)   All axes share ylims
        **plot_kwargs:      (dict)   Parameters passed to 'plot_period'

    Returns:
        Figure and axes objects if return_fig=True, otherwise None
    """
    X = np.array(X)
    assert X.ndim == 3
    n_periods = len(X)
    period_length = X.shape[1]
    X = np.concatenate(X, axis=0)

    # Plot period data
    fig, axes = plot_period(X, channel_names=channel_names, return_fig=True, **plot_kwargs)
    x_sepparations = [(len(X)//n_periods) * i for i in range(1, n_periods)]
    transform = transforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    if y is not None:
        for ind, stage in enumerate(y):
            axes[0].annotate(
                text=str(stage),
                xy=(period_length * ind + (period_length//2), 1.02),
                xycoords=transform,
                ha="center",
                va="bottom"
            )
    if highlight_periods:
        for ax in axes:
            for sep in x_sepparations:
                ax.axvline(sep, color='red',
                           linestyle='--',
                           linewidth=1.5)
    if equal_y_lims:
        set_equal_ylims(axes)

    # Set ticks at all period separation points
    axes[1].set_xticks(np.linspace(0, len(X), n_periods+1).astype(np.int))

    if sample_rate is not None:
        # Set seconds on xaxis
        plt.draw()
        axes[1].set_xticklabels([str(int(tick_lab.get_text())//sample_rate) + "s"
                                 for tick_lab in axes[1].get_xticklabels()],
                                rotation=90)

    # Return, save or show the figure
    if not return_fig:
        if out_path:
            fig.savefig(out_path)
        else:
            plt.show()
        plt.close(fig)
    else:
        return fig, axes
