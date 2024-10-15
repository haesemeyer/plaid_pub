import utility
import matplotlib.pyplot as pl
import seaborn as sns
import pandas as pd
from typing import Optional, Dict
import numpy as np


def lineplot(data: pd.DataFrame, y: str, hue: str, x: np.ndarray, x_name: str, boot_fun: np.mean, nboot=1000,
             line_args: Optional[Dict] = None, shade_args: Optional[Dict] = None) -> pl.Figure:
    """
    Seaborn style lineplot function that does not require x-values to be within the dataframe
    :param data: Dataframe with the plot-data
    :param y: Name of the column with y-values within the dataframe
    :param hue: Name of the column to split observations
    :param x: vector of x-values with same length as y-values
    :param x_name: The label of the x-axis
    :param boot_fun: The function to use for bootstrapping
    :param nboot: The number of bootstrap samples to generate
    :param line_args: Additional keyword arguments for the boot-average lineplot
    :param shade_args: Additional keyword arguments for the boot-error shading
    :return: The generated figure object
    """
    if shade_args is None:
        shade_args = {}
    if line_args is None:
        line_args = {}
    if 'alpha' not in shade_args:
        shade_args['alpha'] = 0.4
    hues = np.unique(data[hue])
    fig = pl.figure()
    for i, h in enumerate(hues):
        values = np.vstack(data[y][data[hue] == h])
        boot_variate = utility.boot_data(values, nboot, boot_fun)
        m = np.mean(boot_variate, axis=0)
        e = np.std(boot_variate, axis=0)
        if 'color' not in shade_args:
            pl.fill_between(x, m - e, m + e, color=f"C{i}", **shade_args)
        else:
            pl.fill_between(x, m - e, m + e, **shade_args)
        if 'color' not in line_args:
            pl.plot(x, m, f"C{i}", label=h, **line_args)
        else:
            pl.plot(x, m, label=h, **line_args)
    pl.xlabel(x_name)
    pl.ylabel(f"{y} +/- se")
    pl.legend()
    sns.despine()
    return fig


def plot_paradigm_overview(fish_data: pd.DataFrame) -> pl.Figure:
    """
    Plots a temporal and spatial overview of paradigm power levels
    :param fish_data: The experimental data
    :return: figure object of the plot
    """
    cmap = pl.colormaps["inferno"]
    exp_phase = fish_data["Experiment phase"]
    to_plot = exp_phase == 9  # Plaid phase in both Plaid and Replay
    plot_colors = cmap(fish_data["Laser power"][to_plot] / np.nanmax(fish_data["Laser power"][to_plot]))
    fig, (ax_spat, ax_temp) = pl.subplots(ncols=2)
    ax_spat.scatter(fish_data["X Position"][to_plot], fish_data["Y Position"][to_plot], s=1, color=plot_colors,
                    rasterized=True)
    ax_spat.set_xlabel("X Position [mm]")
    ax_spat.set_ylabel("Y Position [mm]")
    ax_spat.axis('equal')
    ax_temp.scatter(np.arange(np.sum(to_plot))/250, fish_data["Laser power"][to_plot], s=1, color=plot_colors,
                    rasterized=True)
    ax_temp.set_xlabel("Time [s]")
    ax_temp.set_ylabel("Laser power [mW]")
    sns.despine(fig)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    pass
