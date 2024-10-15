"""
Module to analyze receptive fields of fit models
"""

import pandas as pd
import utility as uti
import model_defs as mdf
import h5py
import argparse
from analysis import CheckArgs, boot_plot
import matplotlib as mpl
import numpy as np
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as pl
import seaborn as sns
from os import path
import os
import loading
from processing import predict_temperature
from scipy.stats import wilcoxon
from multiprocessing import Pool


def prepare_rf(dyn) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts tensors to numpy and splits dynamic receptive field into temperature and history part
    :param dyn: j_dynamic
    :return:
        [0]: temperature rf
        [1]: history rf or None in case of maintain model
        [2]: static rf
    """
    dyn = dyn.numpy()
    if dyn.ndim > 2 and dyn.shape[2] > 1:
        return dyn[0, :, 0], dyn[0, :, 1]
    else:
        return dyn[0, :].ravel(), None


def load_rfs_by_name(datafile: h5py.File, behavior_name: str, loc_dynamic: Optional[np.ndarray] = None,
                     loc_static: Optional[np.ndarray] = None) -> Dict:
    """
    Loads all receptive fields belonging to a specified behavioral output
    :param datafile: The datafile from which to load the receptive fields
    :param behavior_name: The behavioral output name
    :param loc_dynamic: Optionally a non-standard location around which to calculate receptive fields
    :param loc_static: Optionally a non-standard location around which to calculate receptive fields
    :return: Dictionary with filter type as top-level keys followed by experiment type keys
    """
    val_behav_names = ["probability"]
    if behavior_name not in val_behav_names:
        raise ValueError(f"behavior_name has to be one of {val_behav_names} not {behavior_name}")
    # set location around which to compute the receptive fields
    if loc_dynamic is not None:
        xb_dyn = loc_dynamic
    else:
        xb_dyn = x_bar_dyn_pf
    if loc_static is not None:
        xb_static = loc_static
    else:
        xb_static = x_bar_stat_pf
    category = "probability"
    res_dict = {"Temperature": {}, "History": {}}
    m = None
    for experiment in ["plaid", "replay", "plaid_control"]:
        grp = datafile[experiment][category]["weights"]
        indices = list(grp.keys())
        n_models = len(indices)
        for m_index in range(n_models):
            mw_group = grp[f"{m_index}"]
            m_weights = uti.modelweights_from_hdf5(mw_group)
            hist_steps = m_weights[0].shape[0]
            if m is None:
                test_inps = np.random.randn(1, hist_steps, 2)
                test_stat_inps = np.random.randn(1, 2)
                m = mdf.get_standard_boutprob_model(hist_steps)
                m(test_inps, test_stat_inps)
            m.set_weights(m_weights)
            rf_dyn = uti.dca_dr(m, xb_dyn, xb_static, False)
            temp, hist = prepare_rf(rf_dyn)
            if experiment not in res_dict["Temperature"]:
                res_dict["Temperature"][experiment] = []
                res_dict["History"][experiment] = []
            res_dict["Temperature"][experiment].append(temp[-25:])
            if hist is not None:
                res_dict["History"][experiment].append(hist[-25:])
        res_dict["Temperature"][experiment] = np.vstack(res_dict["Temperature"][experiment])
        if len(res_dict["History"][experiment]) == 0:
            del res_dict["History"][experiment]
        else:
            res_dict["History"][experiment] = np.vstack(res_dict["History"][experiment])
    return res_dict


def plot_receptive_fields(res_dict: Dict, s: uti.InputDataStandard) -> pl.Figure:
    fig, axes = pl.subplots(ncols=2, nrows=1, figsize=(18, 4.8))
    time = np.linspace(-1+1/25, 0, 25)
    # Plot Temperature receptive field
    boot_plot(time, res_dict["Temperature"]["plaid"]/s.stimulus_std, 'C0', axes[0], "Plaid")
    boot_plot(time, res_dict["Temperature"]["replay"]/s.stimulus_std, 'C1', axes[0], "Replay")
    boot_plot(time, res_dict["Temperature"]["plaid_control"]/s.stimulus_std, 'C2', axes[0], "Plaid Control")
    axes[0].set_xlabel("Time [s]")
    axes[0].legend()
    # Plot History receptive field if present
    if len(res_dict["History"]) > 0:
        boot_plot(time, res_dict["History"]["plaid"]/s.bout_end_std, 'C0', axes[1], "Plaid")
        boot_plot(time, res_dict["History"]["replay"]/s.bout_end_std, 'C1', axes[1], "Replay")
        boot_plot(time, res_dict["History"]["plaid_control"]/s.bout_end_std, 'C2', axes[1], "Plaid Control")
        axes[1].set_xlabel("Time [s]")
    return fig


def compute_rf_effect(res_dict: Dict, std_stimuli: List[np.ndarray], norm_fun: callable, n_bins=100) -> Tuple[
                                                                                                            np.ndarray,
                                                                                                            np.ndarray,
                                                                                                            np.ndarray,
                                                                                                          pd.DataFrame]:
    """
    Computes the predicted effect the temperature receptive field has on behavior across experiments
    :param res_dict: The receptive field dictionary
    :param std_stimuli: The standardized experimental stimuli
    :param n_bins: The number of bins
    :param norm_fun: Function to convert the network output into the proper behavioral quantity
    :return:
        [0]: The n_bins bin centers
        [1]: n_experiments x n_bins matrix of effect histograms for plaid
        [2]: n_experiments x n_bins matrix of effect histograms for replay
        [3]: Dataframe of effect standard deviations
    """
    std_plaid = []
    std_replay = []
    plaid_effects = []
    replay_effects = []
    for s in std_stimuli:
        pe = np.correlate(s, np.mean(res_dict["Temperature"]["plaid"], 0))[:s.size]
        pe = norm_fun(pe)
        plaid_effects.append(pe)
        re = np.correlate(s, np.mean(res_dict["Temperature"]["replay"], 0))[:s.size]
        re = norm_fun(re)
        replay_effects.append(re)
    all_vals = np.r_[np.hstack(plaid_effects), np.hstack(replay_effects)]
    max_val = np.nanmax(all_vals)
    min_val = np.nanmin(all_vals)
    bins = np.linspace(min_val, max_val, n_bins+1)
    bin_centers = bins[:-1] + np.diff(bins)/2
    plaid_effects_hist = []
    replay_effects_hist = []
    for pe, re in zip(plaid_effects, replay_effects):
        plaid_effects_hist.append(np.histogram(pe, bins, density=True)[0])
        std_plaid.append(np.std(pe))
        replay_effects_hist.append(np.histogram(re, bins, density=True)[0])
        std_replay.append(np.std(re))
    return (bin_centers, np.vstack(plaid_effects_hist), np.vstack(replay_effects_hist),
            pd.DataFrame(np.c_[std_plaid, std_replay], columns=["Plaid", "Replay"]))


def load_stim(e_file: str, std: uti.InputDataStandard) -> np.ndarray:
    fish = loading.load_exp_data_by_info(e_file)[0]
    l_power = np.array(fish["Laser Power"])
    phase = np.array(fish["Experiment phase"])
    l_power = uti.mean_fill_nan(l_power)
    temperature = predict_temperature(l_power / 1000, 1 / 250)[phase == 9]
    # bin down to 25 Hz, i.e. the frequency of our network inputs
    temperature = uti.mean_bin_1d(temperature, 10)
    # return standardized version as expected by networks and therefore our filters
    return (temperature - std.stimulus_mean) / std.stimulus_std


if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="rf_analysis",
                                       description="Analyzes receptive fields of fit models")
    a_parser.add_argument("-f", "--file", help="Path to file with model weights", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-pd", "--p_directory", help="Path to folder with Plaid experiments", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    mw_filepath = args.file
    plaid_dir = args.p_directory

    plot_dir = "plots_receptive_fields"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Organize receptive field extraction in the following manner:
    # For each behavioral output (prob, maintain, distance, magnitude)
    # generate one separate dictionary.
    # The top-level keys in this dictionary will be filter type (temperature, history, static)
    # At these there will be a nested dictionary with the keys (Plaid, Replay, Plaid Control)
    # which will then hold the actualy receptive fields as a n_reps * size matrix
    # The advantage is that this matrix can immediately be used for plotting via bootplot etc.

    with h5py.File(mw_filepath, 'r') as mw_file:
        # load probablity and feature data averages
        x_bar_dyn_pf = mw_file["data_averages_prob"]["x_bar_dyn"][()]
        x_bar_stat_pf = mw_file["data_averages_prob"]["x_bar_stat"][()]
        # load data standards
        standardization = uti.InputDataStandard.from_hdf5(mw_file["standardization"])
        # # create dynamic inputs with "typical" history - re-evaluate: This is a weird point for calculating the
        # history receptive field while it might work well for the temperature one...
        # typ_bout_end = np.zeros(x_bar_dyn_pf.shape[1])
        # typ_bout_end[-21] = 1  # one bout ended 840 ms ago
        # typ_bout_end -= standardization.bout_end_mean
        # typ_bout_end /= standardization.bout_end_std
        # typ_x_bar_dyn_pf = x_bar_dyn_pf.copy()
        # typ_x_bar_dyn_pf[0, :, 1] = typ_bout_end
        # load model receptive fields
        prob_dict = load_rfs_by_name(mw_file, "probability")

    fig = plot_receptive_fields(prob_dict, standardization)
    fig.savefig(path.join(plot_dir, "RF_probability.pdf"))

    # load stimuli from the plaid experiments (NOTE: Replay stimuli are exactly the same by construction)
    load_pool = Pool(10)
    plaid_exps = loading.find_all_exp_paths(plaid_dir)
    ar = [load_pool.apply_async(load_stim, [e, standardization]) for e in plaid_exps]
    all_stimuli = [a.get() for a in ar]
    load_pool.close()

    # Functions to transform receptive field effects into the desired quantities
    def nf_prob(data: np.ndarray) -> np.ndarray:
        pbout_logit = np.log(standardization.bout_end_mean / (1 - standardization.bout_end_mean))
        return 1 / (np.exp(-1*data - pbout_logit) + 1) * 25 - standardization.bout_end_mean * 25

    bc_freq, effect_p_prob, effect_r_prob, std_prob = compute_rf_effect(prob_dict, all_stimuli, nf_prob)

    # Plot histograms of receptive field effects
    fig, ax = pl.subplots()
    boot_plot(bc_freq, effect_p_prob, "C0", ax, "Plaid")
    boot_plot(bc_freq, effect_r_prob, "C1", ax, "Replay")
    pl.xlabel("Delta Bout frequency [Hz]")
    pl.ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "BoutFreq_Temperature_Effect.pdf"))

    # Plot comparisons of standard deviations of receptive field effects (NOTE: This is paired data, each of the two
    # receptive fields is applied to the same stimulus)
    print()
    print("Statistics of differences in receptive field effects.")
    print("Bout probability")
    print(wilcoxon(std_prob["Plaid"], std_prob["Replay"]))
    print()

    fig, ax = pl.subplots()
    ax.scatter(std_prob["Plaid"], std_prob["Replay"], s=2)
    min_val = np.min(np.min(std_prob))
    max_val = np.max(np.max(std_prob))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.set_xlabel("Plaid BF sd [Hz]")
    ax.set_ylabel("Replay BF sd [Hz]")
    sns.despine(fig, ax)
    fig.savefig(path.join(plot_dir, "BoutFreq_SD_Comparison.pdf"))
