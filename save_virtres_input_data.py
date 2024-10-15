"""
Script to analyze difference between Plaid and Replay experiments
"""

import argparse
from typing import Any, List, Tuple, Optional
import os
from os import path
import numpy as np
import pandas as pd
import trainers
import loading
import processing
import multiprocessing as mp
import h5py
from datetime import datetime
import utility
from fit_models import CheckArgs, load_all


def get_data(fish: pd.DataFrame, bout: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stimulus = np.array(fish["Temperature"])
    phase = np.array(fish["Experiment phase"])
    # bin down to 5 Hz
    stim_binned = utility.mean_bin_1d(stimulus, 250//5)
    phase_binned = utility.max_bin_1d(phase, 250//5)

    starts = np.zeros(stim_binned.size)
    binned_bout_start_ix = (np.array(bout["Start"]) // 50).astype(int)
    starts[binned_bout_start_ix] = 1  # NOTE: If our binning is correct this should work

    displacement = np.zeros(stim_binned.size)
    displacement[binned_bout_start_ix] = bout["Displacement"]

    dangle = np.zeros(stim_binned.size)
    dangle[binned_bout_start_ix] = bout["Angle change"]

    return (stim_binned[phase_binned == 9], starts[phase_binned == 9], displacement[phase_binned == 9],
            dangle[phase_binned == 9])


if __name__ == '__main__':

    a_parser = argparse.ArgumentParser(prog="fit_models",
                                       description="Fits MINE models on plaid and replay data storing model weights")
    a_parser.add_argument("-pf", "--p_folder", help="Path to folder with Plaid experiments", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-rf", "--r_folder", help="Path to folder with Replay experiments", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    plaid_folder = args.p_folder
    replay_folder = args.r_folder

    plaid_exps = loading.find_all_exp_paths(plaid_folder)
    replay_exps = loading.find_all_exp_paths(replay_folder)

    # create parallel pool
    load_pool = mp.Pool(10)

    print(f"{len(plaid_exps)} Plaid experiments")
    print(f"{len(replay_exps)} Replay experiments")

    current_date = datetime.now()
    save_dir = f"virt_res_inputs_{current_date.year}_{current_date.month}_{current_date.day}"
    if not path.exists(save_dir):
        os.makedirs(save_dir)

    # load and process all data
    # ar_p = train_pool.apply_async(load_all, [plaid_exps])
    # ar_r = train_pool.apply_async(load_all, [replay_exps])
    # all_plaid_fish, all_plaid_bouts = ar_p.get()
    # all_repl_fish, all_repl_bouts = ar_r.get()
    ar_p = [load_pool.apply_async(load_all, [[pe]]) for pe in plaid_exps]
    ar_r = [load_pool.apply_async(load_all, [[pr]]) for pr in replay_exps]
    all_plaid = [a.get() for a in ar_p]
    all_plaid_fish = [p[0][0] for p in all_plaid]
    all_plaid_bouts = [p[1][0] for p in all_plaid]
    all_replay = [a.get() for a in ar_r]
    all_repl_fish = [r[0][0] for r in all_replay]
    all_repl_bouts = [r[1][0] for r in all_replay]

    # all variables - only needed to calculate normalizing statistics later
    all_starts = []
    all_displacement = []
    all_dangle = []
    all_stimuli = []

    with h5py.File(path.join(save_dir, "plaid_class_model.hdf5"), 'w') as dfile:
        # process and store all plaid information
        for i in range(len(plaid_exps)):
            # build predictors that are needed for our network at 25 Hz
            # stimulus, bout-starts, bout-distances, bout-angles
            stim, st, disp, ang = get_data(all_plaid_fish[i], all_plaid_bouts[i])
            all_stimuli.append(stim)
            all_starts.append(st)
            all_displacement.append(disp)
            all_dangle.append(ang)

            grp = dfile.create_group(f"Plaid_{i}")
            grp.create_dataset("Stimulus", data=stim)
            grp.create_dataset("Starts", data=st)
            grp.create_dataset("Displacement", data=disp)
            grp.create_dataset("DAngle", data=ang)

        for i in range(len(replay_exps)):
            # build predictors that are needed for our network at 25 Hz
            # stimulus, bout-starts, bout-distances, bout-angles
            stim, st, disp, ang = get_data(all_repl_fish[i], all_repl_bouts[i])
            all_stimuli.append(stim)
            all_starts.append(st)
            all_displacement.append(disp)
            all_dangle.append(ang)

            grp = dfile.create_group(f"Replay_{i}")
            grp.create_dataset("Stimulus", data=stim)
            grp.create_dataset("Starts", data=st)
            grp.create_dataset("Displacement", data=disp)
            grp.create_dataset("DAngle", data=ang)

        # add averages for standardization
        dfile.create_dataset("stim_mean", data=np.mean(np.hstack(all_stimuli)))
        dfile.create_dataset("stim_std", data=np.std(np.hstack(all_stimuli)))
        dfile.create_dataset("bs_mean", data=np.mean(np.hstack(all_starts)))
        dfile.create_dataset("bs_std", data=np.std(np.hstack(all_starts)))
        dfile.create_dataset("disp_mean", data=np.mean(np.hstack(all_displacement)))
        dfile.create_dataset("disp_std", data=np.std(np.hstack(all_displacement)))
        dfile.create_dataset("ang_mean", data=np.mean(np.hstack(all_dangle)))
        dfile.create_dataset("ang_std", data=np.std(np.hstack(all_dangle)))
