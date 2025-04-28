import argparse
import os
from os import path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
import loading
import matplotlib as mpl
import utility
from fit_models import CheckArgs, load_all
import pandas as pd
from typing import Dict, Tuple, List
from multiprocessing import Pool


def add_fish_data(b_dict: Dict, paradigm: str, bout: pd.DataFrame, rest_norm: bool) -> None:
    """
    Adds the data from one experiment to the analysis dictionary
    :param b_dict: The overall analysis dictionary
    :param paradigm: The name of the paradigm this experiment belongs to
    :param bout: The bout dataframe of the experiment
    :param rest_norm: Whether to normalize all data by the rest phase values
    """
    phase_seconds = 1200
    phase_rest = 1
    phase_plaid = 9
    phases = bout["Experiment phase"]
    b_dict["Paradigm"] += [paradigm, paradigm]
    b_dict["Phase"] += ["Pre", "Stimulus"]
    disp_rest = bout["Displacement"][phases == phase_rest].mean()
    vig_rest = bout["Average vigor"][phases == phase_rest].mean()
    tm_rest = np.rad2deg(np.mean(np.abs(bout["Angle change"][phases == phase_rest])))
    bf_rest = np.sum(phases == phase_rest) / phase_seconds
    if rest_norm:
        b_dict["Displacement [mm]"] += [1, bout["Displacement"][phases == phase_plaid].mean()/disp_rest]
        b_dict["Tail vigor [deg]"] += [1, bout["Average vigor"][phases == phase_plaid].mean()/vig_rest]
        b_dict["Turn magnitude [deg]"].append(1)
        b_dict["Turn magnitude [deg]"].append(np.rad2deg(np.mean(np.abs(bout["Angle change"][phases ==
                                                                                             phase_plaid])))/tm_rest)
        b_dict["Bout frequency [Hz]"] += [1, np.sum(phases == phase_plaid) / phase_seconds/bf_rest]
    else:
        b_dict["Displacement [mm]"] += [disp_rest, bout["Displacement"][phases == phase_plaid].mean()]
        b_dict["Tail vigor [deg]"] += [vig_rest, bout["Average vigor"][phases == phase_plaid].mean()]
        b_dict["Turn magnitude [deg]"].append(tm_rest)
        b_dict["Turn magnitude [deg]"].append(np.rad2deg(np.mean(np.abs(bout["Angle change"][phases == phase_plaid]))))
        b_dict["Bout frequency [Hz]"] += [bf_rest, np.sum(phases == phase_plaid) / phase_seconds]


def compute_inbout_deltatemps(fish_structure: pd.DataFrame,
                              bout_structure: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the delta-temperature experienced within swim-bouts during the stimulus phase as well as in a shuffled
    control.
    :param fish_structure: Fish dataframe
    :param bout_structure: Bout dataframe
    :return:
        [0]: For each bout the real delta-temperature
        [1]: After rotation of the stimulus the resulting delta-temperatures for each "bout"
    """
    temperatures = fish_structure["Temperature"]
    phases = fish_structure["Experiment phase"]
    # rotate temperatures in stimulus phase then put in the correct position
    rot_temperatures = temperatures.copy()
    stim_temps = temperatures[phases == 9]
    rot_stim_temps = np.roll(stim_temps, np.random.randint(250, 25000))
    rot_temperatures[phases == 9] = rot_stim_temps
    bout_phases = bout_structure["Experiment phase"]
    # our outputs
    real_deltas = np.full(np.sum(bout_phases == 9), np.nan)
    rotated_deltas = np.full(np.sum(bout_phases == 9), np.nan)
    bstarts = np.array(bout_structure["Start"])[bout_phases == 9].astype(int)
    bends = np.array(bout_structure["Stop"])[bout_phases == 9].astype(int)
    for i, (s, e) in enumerate(zip(bstarts, bends)):
        real_deltas[i] = np.abs(temperatures[e] - temperatures[s])
        rotated_deltas[i] = np.abs(rot_temperatures[e] - rot_temperatures[s])
    return real_deltas, rotated_deltas


def weighted_histogram(bins: np.ndarray, bin_data: np.ndarray, weight_data: np.ndarray) -> np.ndarray:
    """
    Calculates the average of a quantity for each bin of another quantity
    :param bins: The bins to consider
    :param bin_data: The data by which to bin
    :param weight_data: The data for which to calculate the average
    :return: The average for each bin
    """
    w_c = np.histogram(bin_data, bins, weights=weight_data)[0].astype(float)
    n_c = np.histogram(bin_data, bins)[0].astype(float)
    return w_c / n_c


def fft_autocorr(x):
    r2 = np.fft.ifft(np.abs(np.fft.fft(x)) ** 2).real
    c = (r2 / x.shape - np.mean(x) ** 2) / np.std(x) ** 2
    return c[:len(x) // 2]


if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="analysis",
                                       description="Runs analysis for high-level comparison of Plaid and Replay"
                                                   " experiments")
    a_parser.add_argument("-pf", "--p_folder", help="Path to folder with Plaid experiments", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-rf", "--r_folder", help="Path to folder with Replay experiments", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    plaid_folder = args.p_folder
    replay_folder = args.r_folder

    plaid_exps = loading.find_all_exp_paths(plaid_folder)
    replay_exps = loading.find_all_exp_paths(replay_folder)

    plot_dir = "overview_plots"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # load and process all data
    load_pool = Pool(10)
    ar_p = [load_pool.apply_async(load_all, [[pe]]) for pe in plaid_exps]
    ar_r = [load_pool.apply_async(load_all, [[pr]]) for pr in replay_exps]
    all_plaid = [a.get() for a in ar_p]
    all_plaid_fish = [p[0][0] for p in all_plaid]
    all_plaid_bouts = [p[1][0] for p in all_plaid]
    all_replay = [a.get() for a in ar_r]
    all_repl_fish = [r[0][0] for r in all_replay]
    all_repl_bouts = [r[1][0] for r in all_replay]

    # for both experiment types plot temperature according to arena position
    # and overlay example trajectory which is also plotted below
    # plot example stimulus and behavior trajectory at same time across plaid, replay and plaid control
    # NOTE: Since frame drops occur at random, we need to align trajectories based on the original frame
    exp_ix = 3
    length = 1100
    s_frame = 600_000
    e_frame = s_frame + length
    plaid_orig_frame_index = all_plaid_fish[exp_ix]["TFile frame"][s_frame]
    s_frame_repl = np.where(all_repl_fish[exp_ix]["TFile frame"] == plaid_orig_frame_index)[0][0]
    e_frame_repl = s_frame_repl + length

    cmap = pl.colormaps["inferno"]
    all_x_plaid = np.hstack([pf["X Position"][pf["Experiment phase"] == 9] for pf in all_plaid_fish])
    selector = np.random.rand(all_x_plaid.size)
    selected = selector < 0.1
    all_x_plaid = all_x_plaid[selected]
    all_y_plaid = np.hstack([pf["Y Position"][pf["Experiment phase"] == 9] for pf in all_plaid_fish])[selected]
    all_t_plaid = np.hstack([pf["Temperature"][pf["Experiment phase"] == 9] for pf in all_plaid_fish])[selected]
    fig, (ax_plot, ax_cbar) = pl.subplots(ncols=2, width_ratios=[1, 0.1])
    sc = ax_plot.scatter(all_x_plaid, all_y_plaid, s=1, c=all_t_plaid, rasterized=True, cmap=cmap)
    ax_plot.plot(all_plaid_fish[exp_ix]["X Position"][s_frame:e_frame],
                 all_plaid_fish[exp_ix]["Y Position"][s_frame:e_frame], 'w')
    ax_plot.plot(all_plaid_fish[exp_ix]["X Position"][s_frame],
                 all_plaid_fish[exp_ix]["Y Position"][s_frame], 'w.')
    ax_plot.set_xlabel("X Position [mm]")
    ax_plot.set_ylabel("Y Position [mm]")
    ax_plot.axis('equal')
    pl.colorbar(sc, cax=ax_cbar)
    fig.savefig(path.join(plot_dir, f"Plaid_Temp_By_Position.pdf"), dpi=600)

    all_x_repl = np.hstack([rf["X Position"][rf["Experiment phase"] == 9] for rf in all_repl_fish])
    selector = np.random.rand(all_x_repl.size)
    selected = selector < 0.1
    all_x_repl = all_x_repl[selected]
    all_y_repl = np.hstack([rf["Y Position"][rf["Experiment phase"] == 9] for rf in all_repl_fish])[selected]
    all_t_repl = np.hstack([rf["Temperature"][rf["Experiment phase"] == 9] for rf in all_repl_fish])[selected]
    fig, (ax_plot, ax_cbar) = pl.subplots(ncols=2, width_ratios=[1, 0.1])
    sc = ax_plot.scatter(all_x_repl, all_y_repl, s=1, c=all_t_repl, rasterized=True, cmap=cmap)
    ax_plot.plot(all_repl_fish[exp_ix]["X Position"][s_frame_repl:e_frame_repl],
                 all_repl_fish[exp_ix]["Y Position"][s_frame_repl:e_frame_repl], 'w')
    ax_plot.plot(all_repl_fish[exp_ix]["X Position"][s_frame_repl],
                 all_repl_fish[exp_ix]["Y Position"][s_frame_repl], 'w.')
    ax_plot.set_xlabel("X Position [mm]")
    ax_plot.set_ylabel("Y Position [mm]")
    ax_plot.axis('equal')
    pl.colorbar(sc, cax=ax_cbar)
    fig.savefig(path.join(plot_dir, f"Repl_Temp_By_Position.pdf"), dpi=600)

    # plot temperature stimulus auto-correlation

    all_acorr = np.vstack([fft_autocorr(pf["Temperature"][pf["Experiment phase"] == 9])[:250*10]
                           for pf in all_plaid_fish])
    acorr_time = np.arange(all_acorr.shape[1]) / 250
    m = np.mean(all_acorr, axis=0)
    s = np.std(all_acorr, axis=0)

    fig = pl.figure()
    pl.fill_between(acorr_time, m-s, m+s, color='k', alpha=0.5)
    pl.plot(acorr_time, m, color='k')
    pl.xlabel("Time [s]")
    pl.ylabel("Autocorrelation")
    sns.despine()
    fig.savefig(path.join(plot_dir, f"Stimulus_AutoCorrelation.pdf"))


    time = np.linspace(0, length/250, length, endpoint=False)
    fig, axes = pl.subplots(nrows=4, ncols=2, sharey='row', sharex='all', figsize=[16, 9])
    ax_plaid = axes[:, 0]
    ax_repl = axes[:, 1]

    ax_plaid[0].plot(time, all_plaid_fish[exp_ix]["Temperature"][s_frame:e_frame], 'k')
    ax_plaid[0].set_ylabel("Stimulus [C]")
    ax_plaid[1].plot(time, all_plaid_fish[exp_ix]["Instant speed"][s_frame:e_frame], 'k')
    ax_plaid[1].set_ylabel("Speed [mm/s]")
    ax_plaid[2].plot(time, np.rad2deg(all_plaid_fish[exp_ix]["Tail tip angle"][s_frame:e_frame]), 'k')
    ax_plaid[2].set_ylabel("Tail angle [degree]")
    ax_plaid[3].plot(time, np.rad2deg(all_plaid_fish[exp_ix]["Heading"][s_frame:e_frame]), 'k')
    ax_plaid[3].set_xlabel("Time [s]")
    ax_plaid[3].set_ylabel("Heading [degree]")

    ax_repl[0].plot(time, all_repl_fish[exp_ix]["Temperature"][s_frame_repl:e_frame_repl], 'k')
    ax_repl[1].plot(time, all_repl_fish[exp_ix]["Instant speed"][s_frame_repl:e_frame_repl], 'k')
    ax_repl[2].plot(time, np.rad2deg(all_repl_fish[exp_ix]["Tail tip angle"][s_frame_repl:e_frame_repl]), 'k')
    ax_repl[3].plot(time, np.rad2deg(all_repl_fish[exp_ix]["Heading"][s_frame_repl:e_frame_repl]), 'k')
    ax_repl[3].set_xlabel("Time [s]")

    sns.despine()
    fig.tight_layout()
    fig.savefig(path.join(plot_dir, f"Example_StimAndBehavior.pdf"))

    # for experimental phase plot distributions of:
    # 1) interbout intervals
    # 2) bout displacements
    # 3) turn angles
    # 4) frequency of turn flip and turn maintenance streak lengths vs expected frequencies
    #   The null-hypothesis would be that turn direction at time t is independent of direction at time t-1
    #   In that case the chance of observing a maintain streak of length N or a an alternating streak of length M
    #   when starting at a random timepoint t, assuming that p(left)=p(right)=0.5 would be
    #   P(N_maintain) = p(left)^N * p(right) + p(right)^N * p(left) = 2 * 0.5^(N+1) = 0.5^N | N>=1
    #   P(M_alternate) = p(left)*p(right)*p(left)*... + p(right)*p(left)*p(right)*... = 0.5^M + 0.5^M = 0.5^(M-1) | M>=2
    #   Note the apparent glitch here: The probabilities of all maintain and all alternate streaks sum up to 1 each
    #   even though they should be exclusive (impossible to maintain and alternate at the same time)
    #   This is perhaps explained by "maintain" containing the 1-streak which is alternating and "alternate" not looking
    #   ahead post the end of the streak, i.e. the last element of an M-alternate streak could be the start of any
    #   length N maintain streak. Specifically, if requiring N>=2 maintain would sum up to 0.5 and likewise for
    #   alternate if requiring the next element (M+1) to still be an alternation
    #   Just considering P(N_maintain) [by symmetrie the same argument can be made for alternation], the overall
    #   probability to maintain turn direction, if it is a homogeneous process, is 1-P(1). This can be used to calculate
    #   expected frequencies of turn streaks, calling P(1_maintain) 1-q:
    #   P(N) = (1-q)*q^(N-1) | N>=1
    #   Importantly, if the process is inhomogeneous the real frequencies  won't line up with the expectation
    #   since (P(N|q1) + P(N|q2))/2 != P(N|(q1+q2)/2) except for N=1
    #   This means for each experiment, we can compute q

    def compute_distr_stats(name: str) -> None:
        all_p = [b[name][b["Experiment phase"] == 9] for b in all_plaid_bouts]
        all_r = [b[name][b["Experiment phase"] == 9] for b in all_repl_bouts]
        all_bline = [b[name][b["Experiment phase"] == 1] for b in (all_plaid_bouts+all_repl_bouts)]
        pv, _, stat = utility.ks_bootstrap_test_by_fish(all_p, all_r, 10000)
        print(f"{name} Plaid vs. Replay: p={pv}; stat={stat}; N={len(all_p) + len(all_r)}")

        pv, _, stat = utility.ks_bootstrap_test_by_fish(all_p, all_bline, 10000)
        print(f"{name} Plaid vs. Baseline: p={pv}; stat={stat}; N={len(all_p) + len(all_bline)}")

        pv, _, stat = utility.ks_bootstrap_test_by_fish(all_r, all_bline, 10000)
        print(f"{name} Replay vs. Baseline: p={pv}; stat={stat}; N={len(all_bline) + len(all_r)}")

    # def ks_bootstrap_test_by_bout(sample1: List, sample2: List, nboot: int) -> Tuple[float, np.ndarray, float]:
    #     true_1 = np.hstack(sample1)
    #     true_2 = np.hstack(sample2)
    #     true_ks_stat = kstest(true_1, true_2).statistic
    #     combined = np.r_[true_1, true_2]
    #     ix_combined = np.arange(combined.size).astype(int)
    #     boot_stats = np.zeros(nboot)
    #     for i in range(nboot):
    #         ix1 = np.random.choice(ix_combined, true_1.size)
    #         ix2 = np.random.choice(ix_combined, true_2.size)
    #         s1 = combined[ix1]
    #         s2 = combined[ix2]
    #         boot_stats[i] = kstest(s1, s2).statistic
    #     # estimate p-value based on normal approximation if no elements in boot_stats are larger than true_ks_stat
    #     if np.sum(boot_stats >= true_ks_stat) == 0:
    #         print("p-value estimated")
    #         std = np.std(boot_stats)
    #         n_sigma = (true_ks_stat - np.mean(boot_stats)) / std
    #         p = 1 - norm.cdf(n_sigma)
    #     else:
    #         p = np.sum(boot_stats >= true_ks_stat) / nboot
    #     return p, boot_stats, true_ks_stat


    ibi_bins = np.linspace(0, 3000, 151)
    ibi_bc = ibi_bins[:-1] + np.diff(ibi_bins)/2
    all_ibi_hist_p = np.vstack([np.histogram(b["IBI"][b["Experiment phase"] == 9], bins=ibi_bins, density=True)[0] for b in all_plaid_bouts])
    all_ibi_hist_r = np.vstack([np.histogram(b["IBI"][b["Experiment phase"] == 9], bins=ibi_bins, density=True)[0] for b in all_repl_bouts])
    all_ibi_hist_bline = np.vstack([np.histogram(b["IBI"][b["Experiment phase"] == 1], bins=ibi_bins, density=True)[0] for b in (all_repl_bouts+all_plaid_bouts)])

    fig = pl.figure()
    m, e = utility.boot_error(all_ibi_hist_p, 1000, np.mean)
    pl.fill_between(ibi_bc, m-e, m+e, color='C0', alpha=0.3)
    pl.plot(ibi_bc, m, 'C0', label="Plaid")
    m, e = utility.boot_error(all_ibi_hist_r, 1000, np.mean)
    pl.fill_between(ibi_bc, m - e, m + e, color='C3', alpha=0.3)
    pl.plot(ibi_bc, m, 'C3', label="Replay")
    m, e = utility.boot_error(all_ibi_hist_bline, 1000, np.mean)
    pl.fill_between(ibi_bc, m - e, m + e, color='k', alpha=0.3)
    pl.plot(ibi_bc, m, 'k--', label="Baseline")
    pl.xlabel("Interbout interval [ms]")
    pl.ylabel("Density")
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "IBI_Distribution.pdf"))


    disp_bins = np.linspace(0, 10, 250)
    disp_bc = disp_bins[:-1] + np.diff(disp_bins)/2
    all_disp_hist_p = np.vstack([np.histogram(b["Displacement"][b["Experiment phase"] == 9], bins=disp_bins, density=True)[0] for b in all_plaid_bouts])
    all_disp_hist_r = np.vstack([np.histogram(b["Displacement"][b["Experiment phase"] == 9], bins=disp_bins, density=True)[0] for b in all_repl_bouts])
    all_disp_hist_bline = np.vstack([np.histogram(b["Displacement"][b["Experiment phase"] == 1], bins=disp_bins, density=True)[0] for b in (all_repl_bouts+all_plaid_bouts)])

    fig = pl.figure()
    m, e = utility.boot_error(all_disp_hist_p, 1000, np.mean)
    pl.fill_between(disp_bc, m - e, m + e, color='C0', alpha=0.3)
    pl.plot(disp_bc, m, 'C0', label="Plaid")
    m, e = utility.boot_error(all_disp_hist_r, 1000, np.mean)
    pl.fill_between(disp_bc, m - e, m + e, color='C3', alpha=0.3)
    pl.plot(disp_bc, m, 'C3', label="Replay")
    m, e = utility.boot_error(all_disp_hist_bline, 1000, np.mean)
    pl.fill_between(disp_bc, m - e, m + e, color='k', alpha=0.3)
    pl.plot(disp_bc, m, 'k--', label="Baseline")
    pl.xlabel("Displacement [mm]")
    pl.ylabel("Density")
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "Displacement_Distribution.pdf"))

    ang_bins = np.linspace(-180, 180, 250)
    ang_bc = ang_bins[:-1] + np.diff(ang_bins)/2
    all_ang_hist_p = np.vstack([np.histogram(np.rad2deg(b["Angle change"][b["Experiment phase"] == 9]), bins=ang_bins, density=True)[0] for b in all_plaid_bouts])
    all_ang_hist_r = np.vstack([np.histogram(np.rad2deg(b["Angle change"][b["Experiment phase"] == 9]), bins=ang_bins, density=True)[0] for b in all_repl_bouts])
    all_ang_hist_bline = np.vstack([np.histogram(np.rad2deg(b["Angle change"][b["Experiment phase"] == 9]), bins=ang_bins, density=True)[0] for b in (all_repl_bouts + all_plaid_bouts)])

    fig = pl.figure()
    m, e = utility.boot_error(all_ang_hist_p, 1000, np.mean)
    pl.fill_between(ang_bc, m - e, m + e, color='C0', alpha=0.3)
    pl.plot(ang_bc, m, 'C0', label="Plaid")
    m, e = utility.boot_error(all_ang_hist_r, 1000, np.mean)
    pl.fill_between(ang_bc, m - e, m + e, color='C3', alpha=0.3)
    pl.plot(ang_bc, m, 'C3', label="Replay")
    m, e = utility.boot_error(all_ang_hist_bline, 1000, np.mean)
    pl.fill_between(ang_bc, m - e, m + e, color='k', alpha=0.3)
    pl.plot(ang_bc, m, 'k--', label="Baseline")
    pl.xlabel("Turn angle [degrees]")
    pl.ylabel("Density")
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "Turn_Distribution.pdf"))

    compute_distr_stats("IBI")
    compute_distr_stats("Displacement")
    compute_distr_stats("Angle change")

    all_tdir_p = [np.array(np.sign(b["Angle change"][b["Experiment phase"] == 9]) > 0) for b in all_plaid_bouts]
    all_tdir_r = [np.array(np.sign(b["Angle change"][b["Experiment phase"] == 9]) > 0) for b in all_repl_bouts]

    streak_lengths = np.arange(10) + 1

    p_streak_p = np.vstack([utility.count_turn_streaks(tdp, 10)[0]/tdp.size for tdp in all_tdir_p])
    b_streak_p = utility.boot_data(p_streak_p, 1000, np.mean)
    m_p = np.mean(b_streak_p, axis=0)
    e_p = np.std(b_streak_p, axis=0)
    p_streak_r = np.vstack([utility.count_turn_streaks(tdr, 10)[0]/tdr.size for tdr in all_tdir_r])
    b_streak_r = utility.boot_data(p_streak_r, 1000, np.mean)
    m_r = np.mean(b_streak_r, axis=0)
    e_r = np.std(b_streak_r, axis=0)
    # if left and right turns were equally distributed our null-hypothesis would be p_maintain=p_switch=0.5
    # to account for fish-specific handedness, create shuffles instead to maintain l/r distribution but to break
    # real increases in p_maintain
    tdir_shuffles = all_tdir_p + all_tdir_r
    rng = np.random.default_rng()
    tdir_shuffles = [rng.permuted(tds) for tds in tdir_shuffles]
    p_streak_exp = np.vstack([utility.count_turn_streaks(tds, 10)[0]/tds.size for tds in tdir_shuffles])
    b_streak_exp = utility.boot_data(p_streak_exp, 1000, np.mean)
    m_exp = np.mean(b_streak_exp, axis=0)
    e_exp = np.std(b_streak_exp, axis=0)

    fig = pl.figure()
    pl.errorbar(streak_lengths, m_p, e_p, label="Plaid", ls='none', marker='.')
    pl.errorbar(streak_lengths, m_r, e_r, label="Replay", ls='none', marker='.')
    pl.errorbar(streak_lengths, m_exp, e_exp, label="Random", ls='none', marker='.', color='k')
    pl.xlabel("Streak length")
    pl.ylabel("Frequency")
    pl.legend()
    pl.yscale('log')
    sns.despine()
    fig.savefig(path.join(plot_dir, "Turn_streak_length_probs.pdf"))

    # compare delta-temperatures experienced by plaid and replay fish during bouts vs. random periods
    d_inbout_dt = {"Experiment": [], "Type": [], "Avg. delta-T": []}
    for f, b in zip(all_plaid_fish, all_plaid_bouts):
        real, rot = compute_inbout_deltatemps(f, b)
        d_inbout_dt["Experiment"] += ["Plaid", "Plaid"]
        d_inbout_dt["Type"] += ["Bout", "Random"]
        d_inbout_dt["Avg. delta-T"] += [np.mean(real), np.mean(rot)]
    for f, b in zip(all_repl_fish, all_repl_bouts):
        real, rot = compute_inbout_deltatemps(f, b)
        d_inbout_dt["Experiment"] += ["Replay", "Replay"]
        d_inbout_dt["Type"] += ["Bout", "Random"]
        d_inbout_dt["Avg. delta-T"] += [np.mean(real), np.mean(rot)]
    df_inbout_dt = pd.DataFrame(d_inbout_dt)

    fig = pl.figure()
    sns.boxplot(data=df_inbout_dt, x="Experiment", y="Avg. delta-T", hue="Type", width=0.4, whis=np.inf)
    sns.despine()
    fig.savefig(path.join(plot_dir, "InBout_DeltaT.pdf"))
