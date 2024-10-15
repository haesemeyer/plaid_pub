"""
Script to analyze models fit on plaid and replay data
"""

import argparse
from typing import Any, List, Tuple
import os
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pl

import matplotlib as mpl
import utility
import h5py
import plot_funs as pf
from fit_models import CheckArgs
from sklearn.metrics import roc_auc_score
from overview_plots import weighted_histogram


def plot_and_save_all(exp_list: List[str], fish_list: List[pd.DataFrame]) -> None:
    """
    Plots paradigm overview for all experiments and saves figure with experiment name indication
    :param exp_list: List of experiment info file paths
    :param fish_list: List of corresponding fish dataframes
    """
    for fpath, fdata in zip(exp_list, fish_list):
        ename = path.split(fpath)[-1][:-5]
        figure = pf.plot_paradigm_overview(fdata)
        figure.savefig(path.join(plot_dir, f"{ename}_overview.pdf"), dpi=600)


def get_prob_test(grp: h5py.Group) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    For probability type experiments gets raw and processed test data
    :param grp: The file group in which individual fit test data are stored
    :return:
        [0]: Vector of ROC AUC for each fit
        [1]: List of true outcomes for each experiment
        [2]: List of predicted outcomes for each experiment
    """
    auc_data = []
    all_true = []
    all_pred = []
    for k in grp:
        true_outcome = grp[k][[key for key in grp[k] if "true" in key][0]][()]
        all_true.append(true_outcome)
        predictions = grp[k][[key for key in grp[k] if "pred" in key][0]][()]
        all_pred.append(predictions)
        auc_data.append(roc_auc_score(true_outcome > 0.5, predictions))
    return np.hstack(auc_data), all_true, all_pred


def boot_plot(xvals: np.ndarray, data: np.ndarray, color: Any, axis: pl.Axes, label: str) -> None:
    bs = utility.boot_data(data, 1000, np.nanmean)
    m = np.mean(bs, 0)
    e = np.std(bs, 0)
    axis.fill_between(xvals, m - e, m + e, color=color, alpha=0.4)
    axis.plot(xvals, m, color=color, label=label)


if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="analysis",
                                       description="Analyze pre-fit models and their test performance")
    a_parser.add_argument("-f", "--file", help="Path to fit model file", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    dfile_path = args.file

    plot_dir = "plots_analysis"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # analyze model test performance

    # for our probability models (boutprob and flip) visualize classification performance
    # both as a bar (?) plot comparing ROC-AUC across individual fits and using a det-curve
    # on the entire concatenated data (sort of the "average det curve")
    conditions = ["naive", "plaid", "plaid_control", "replay"]
    prob_models = ["probability"]

    auc_dict = {"Condition": [], "AUC": [], "Model": []}
    qq_dict = {}
    distr_dict = {}
    prob_bins = np.linspace(0, 1, 200)
    prob_bc = prob_bins[:-1] + np.diff(prob_bins)/2
    dist_bins = np.linspace(0, 8, 200)
    dist_bc = (dist_bins[:-1] + np.diff(dist_bins)/2)
    mag_bins = np.linspace(0, 150, 200)
    mag_bc = mag_bins[:-1] + np.diff(mag_bins) / 2

    with h5py.File(dfile_path, "r") as fit_file:
        standardization = utility.InputDataStandard.from_hdf5(fit_file["standardization"])
        for condition in conditions:
            cond_grp = fit_file[condition]
            for model in prob_models:
                if model not in qq_dict:
                    qq_dict[model] = {}
                model_group = cond_grp[model]
                auc, true, pred = get_prob_test(model_group["test"])
                qq = np.vstack([weighted_histogram(prob_bins, p, t) for p, t in zip(pred, true)])
                qq_dict[model][condition] = qq
                auc_dict["Condition"] += [condition]*auc.size
                auc_dict["Model"] += [model]*auc.size
                auc_dict["AUC"] += [a for a in auc]

    # Plot AUC comparison for probability and maintain models
    fig, ax = pl.subplots()
    sns.boxplot(data=auc_dict, x="Condition", y="AUC", hue="Model", ax=ax, whis=np.inf)
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "Model_AUC.pdf"))

    # Plot QQ comparisons for all models
    fig, ax = pl.subplots()
    boot_plot(prob_bc, qq_dict["probability"]["plaid"], "C0", ax, "Plaid")
    boot_plot(prob_bc, qq_dict["probability"]["replay"], "C1", ax, "Replay")
    boot_plot(prob_bc, qq_dict["probability"]["plaid_control"], "C2", ax, "Plaid Control")
    xmin, xmax = ax.get_xlim()
    ax.plot([0, 0.2], [0, 0.2], 'k--')
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True proportion")
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    pl.legend()
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "Probability_qq_plot.pdf"))
