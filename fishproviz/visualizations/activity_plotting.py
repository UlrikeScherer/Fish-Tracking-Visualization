import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from fishproviz.metrics import metric_names
from fishproviz.metrics.results_to_csv import get_filename_for_metric_csv
from fishproviz.utils import get_date_string
from fishproviz.config import VIS_DIR, sep


colors = (
    np.array(
        [
            (166, 206, 227),
            (31, 120, 180),
            (178, 223, 138),
            (51, 160, 44),
            (251, 154, 153),
            (227, 26, 28),
            (253, 191, 111),
            (255, 127, 0),
            (202, 178, 214),
            (106, 61, 154),
            (255, 255, 153),
            (177, 89, 40),
        ]
    )
    / 255
)

# color map for each of the 24 fishes
color_map = colors  # [colors[int(k/2)] for k in range(colors.shape[0]*2)]


def plot_activity(data, time_interval):
    """Plots the average activity my mean and vaiance over time
    input: data, time_interval
    return: figure
    """
    fig, ax = plt.subplots(figsize=(15 * (data.shape[0] / 300), 5))
    plt.tight_layout()
    offset = int(time_interval / 2)
    ax.errorbar(
        range(offset, offset + len(data) * time_interval, time_interval),
        data[:, 0],
        [data[:, 0], data[:, 1]],
        marker=".",
        linestyle="None",
        elinewidth=0.7,
    )
    ax.set_xlabel("seconds")
    return fig


def sliding_window_figures_for_tex(
    dataset,
    *args,
    fish_keys=None,
    day_keys=None,
    name="methode",
    set_title=False,
    set_legend=False,
    **kwargs
):
    ncols = 6
    if day_keys is None:
        day_keys = list(dataset[fish_keys[0]].keys())
    for i in range(0, 29, ncols):
        f = sliding_window(
            dataset,
            fish_keys=fish_keys,
            day_keys=day_keys[i : i + ncols],
            *args,
            name="%s_%02d-%02d" % (name, i, i + ncols - 1),
            set_title=set_title,
            set_legend=set_legend,
            first_day=i,
            **kwargs
        )
        plt.close(f)
    return None


def sliding_window(
    dataset,
    time_interval,
    sw,
    fish_keys=None,
    day_keys=None,
    fish_labels=None,
    xlabel="seconds",
    ylabel="average cm/Frame",
    name="methode",
    write_fig=False,
    logscale=False,
    baseline=None,
    set_title=True,
    set_legend=True,
):
    """Summerizes the data for a sliding window and plots a continuous line over time"""
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = False

    offset = int(time_interval * sw / 2)
    x_max = offset
    if isinstance(dataset, np.ndarray):
        dataset = [[dataset]]
    n_fishes = len(fish_keys)
    if day_keys is None:
        day_keys = list(dataset[fish_keys[0]].keys())
    n_days = len(day_keys)
    print("Number of fishes:", n_fishes, " Number of days: ", n_days)
    ncols = 6
    nrows = int(np.ceil(n_days / ncols))
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * 6, 4 * nrows), sharey=True
    )
    if n_days == 1:
        axes = [axes]
    if nrows > 1:
        axes = np.ravel(axes)
    fig.tight_layout()
    # color_map = plt.get_cmap('tab20b').colors + plt.get_cmap('tab20b').colors[:4]

    for i, f_key in enumerate(fish_keys):
        for d_idx, d_key in enumerate(day_keys):
            data = dataset[f_key][d_key]
            slide_data = [
                np.mean(data[i : i + sw, 0]) for i in range(0, data.shape[0] - sw + 1)
            ]
            x_end = offset + (len(data) - sw + 1) * time_interval
            x_max = max(x_max, x_end)  # x_max update to draw the dashed baseline
            plot_metric_data(
                axes[d_idx],
                range(offset, x_end, time_interval),
                slide_data,
                linestyle="-",
                label=fish_labels[i],
                color=color_map[i],
                linewidth=2,
            )
            if i == 0:
                if set_title:
                    axes[d_idx].set_title(
                        "Date %s" % get_date_string(d_key), y=0.95, pad=4
                    )
                if logscale:
                    axes[d_idx].set_yscale("log")
                if d_idx >= (nrows - 1) * ncols:
                    axes[d_idx].set_xlabel(xlabel)
                axes[d_idx].grid(axis="y")
                if d_idx % ncols == 0:
                    axes[d_idx].set_ylabel(ylabel, fontsize=20)
    if baseline is not None:
        for i in range(n_days):
            plot_metric_data(
                axes[i],
                (offset, time_interval * ((x_max // time_interval) - 1)),
                (baseline, baseline),
                linestyle=":",
                color="black",
            )

    for i in range(n_days, len(axes)):
        axes[i].axis("off")

    if set_legend:
        set_legend_for_metric_plot(
            axes[0],
            loc="upper center",
            bbox_to_anchor=(ncols / 2 + 0.15, 1.55),
            ncol=n_fishes,
        )

    if write_fig:
        fig.savefig(get_filepath_metric_plot(name, time_interval), bbox_inches="tight")
    return fig


def plot_metric_data(
    ax,
    x,
    y,
    color,
    label=None,
    linestyle="-",
    x_label=None,
    y_label=None,
    title=None,
    linewidth=2,
    **kwargs
):
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs
    )
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y")
    return ax


def set_legend_for_metric_plot(
    ax, legend_loc="upper center", ncol=1, fontsize=18, bbox_to_anchor=(0.5, 1.05)
):
    leg = ax.legend(
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fancybox=True,
        fontsize=fontsize,
        markerscale=2,
    )
    for line in leg.get_lines():
        line.set_linewidth(7.0)
    return ax


def plot_turning_direction(data, time_interval):
    fig, ax = plt.subplots(figsize=(15 * (data.shape[0] / 300), 5))
    plt.tight_layout()
    offset = int(time_interval / 2)
    ax.errorbar(
        range(offset, offset + len(data) * time_interval, time_interval),
        data[:, 0],
        data[:, 1],
        marker=".",
        linestyle="None",
        elinewidth=0.7,
    )
    ax.set_xlabel("seconds")
    return fig


def plot_metric_figure_for_days(metric_name, measure=None, write_fig=True):
    file_e = get_filename_for_metric_csv(
        metric_name, "DAY", False, measure_name=measure
    )
    if not os.path.isfile(file_e):
        print("File not found:", file_e)
        return

    df = pd.read_csv(file_e, sep=sep)
    days = df.columns[2:]
    days = list(map(lambda d: "%s/%s" % (d[4:6], d[6:8]), days))
    df_np = df.to_numpy()

    fig, axis = plt.subplots(figsize=(25, 5), ncols=2, sharey=True)
    nfish = df_np.shape[0]
    ncols = int(np.ceil(nfish / 2))
    for axi, ax in enumerate(axis):
        start = 0 + axi * ncols
        for i in range(start, min(start + ncols, nfish)):
            plot_metric_data(
                ax,
                days,
                df_np[i, 2:],
                color=color_map[i % len(color_map)],
                label=df_np[i, 1][6:10],
                linestyle="--",
                marker="o",
                markersize=2,
                linewidth=0.2,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("%s %s" % (metric_name, measure if measure is not None else ""))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y")
        set_legend_for_metric_plot(
            ax, ncol=ncols, fontsize=8, bbox_to_anchor=(0.5, 1.15)
        )
    if write_fig:
        fig.savefig(
            get_filepath_metric_plot(metric_name, measure=measure), bbox_inches="tight"
        )

def get_filepath_metric_plot(metric_name, measure=None, subdir=None, ext="pdf"):
    dir2plot = "{}/".format(VIS_DIR)
    if subdir is not None:
        dir2plot = "{}/{}/".format(dir2plot, subdir)
    os.makedirs(dir2plot, exist_ok=True)
    return "{}/{}{}.{}".format(
        dir2plot, metric_name, "_" + measure if measure is not None else "", ext
    )

def plots_over_life_time():
    for metric_name in metric_names:
        if metric_name == "entropy":
            plot_metric_figure_for_days(metric_name, measure=None)
        else:
            plot_metric_figure_for_days(metric_name, measure="mean")
            plot_metric_figure_for_days(metric_name, measure="std")


if __name__ == "__main__":
    plots_over_life_time()
