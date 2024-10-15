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


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'p_folder' or self.dest == 'r_folder' or self.dest == "p_directory":
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        elif self.dest == 'file':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified file does not exist")
            if not path.isfile(values):
                raise argparse.ArgumentError(self, "The destination might be a directory but is not a file")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def load_all(exp_list: List[str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    For list of experiments returns pre-processed fish data and bout data frames
    :param exp_list: List of experimental info file paths
    :return:
        [0]: List of pre-processed fish dataframes
        [1]: List of extracted bout dataframes
    """
    all_fish_data = []
    all_bout_data = []
    for e in exp_list:
        fish = loading.load_exp_data_by_info(e)[0]
        processing.pre_process_fish_data(fish)
        all_fish_data.append(fish)
        bouts = processing.identify_bouts(fish, 0.1)
        all_bout_data.append(bouts)
    return all_fish_data, all_bout_data


def train_test(all_fish: List[pd.DataFrame], all_bouts: List[pd.DataFrame], train_fraction: float,
               model_hist: int, tbatch: int, std: utility.InputDataStandard, n_epochs: int, shuffle: bool):
    """
    Train and test models
    :param all_fish: All experiments
    :param all_bouts: All associated bouts
    :param train_fraction: The fraction of frames in each experiment that should be train data
    :param model_hist: Length of the history in the model
    :param tbatch: Size of training batches
    :param std: Input data standardizations
    :param n_epochs: The number of training epochs
    :param shuffle: If set to true, rotate outputs with respect to inputs by 1/3 of the data-length
    :return:
    """
    # Probability model
    prob_trainer = trainers.BoutProbabilityTrainer(model_hist, tbatch, std, all_fish, all_bouts, train_fraction,
                                                   shuffle)
    prob_true_naive, prob_pred_naive = prob_trainer.test()
    prob_trainer.train(n_epochs)
    prob_weights = prob_trainer.get_model_weights()
    prob_true_trained, prob_pred_trained = prob_trainer.test()

    naive_results = {
        'prob_true_naive': prob_true_naive,
        'prob_pred_naive': prob_pred_naive
    }
    trained_results = {
        'prob_true_trained': prob_true_trained,
        'prob_pred_trained': prob_pred_trained,
        'prob_weights': prob_weights
    }
    return naive_results, trained_results


def save_pool_results(pool_results, iteration: int, file_group: h5py.Group, naive_group: Optional[h5py.Group]):
    naive, trained = pool_results.get()
    if naive_group is not None:
        naive_group.create_dataset(f"probability/test/{iteration}/prob_true_naive", data=naive['prob_true_naive'])
        naive_group.create_dataset(f"probability/test/{iteration}/prob_pred_naive", data=naive['prob_pred_naive'])
    file_group.create_dataset(f"probability/test/{iteration}/prob_true_trained", data=trained['prob_true_trained'])
    file_group.create_dataset(f"probability/test/{iteration}/prob_pred_trained", data=trained['prob_pred_trained'])
    utility.modelweights_to_hdf5(file_group.create_group(f"probability/weights/{iteration}"), trained['prob_weights'])


if __name__ == '__main__':

    # if the unix-default "fork" is used we cannot properly set maxtasksperchild in pool creation below
    # therefore force process creation as 'spawn' (windows and osx default)
    mp.set_start_method('spawn')

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

    # create parallel pool - setting max tasks to clean up memory and avoid tensorflow slow-down
    train_pool = mp.Pool(5, maxtasksperchild=1)

    print(f"{len(plaid_exps)} Plaid experiments")
    print(f"{len(replay_exps)} Replay experiments")

    current_date = datetime.now()
    fit_dir = f"model_fits_{current_date.year}_{current_date.month}_{current_date.day}"
    if not path.exists(fit_dir):
        os.makedirs(fit_dir)

    # load and process all data
    ar_p = [train_pool.apply_async(load_all, [[pe]]) for pe in plaid_exps]
    ar_r = [train_pool.apply_async(load_all, [[pr]]) for pr in replay_exps]
    all_plaid = [a.get() for a in ar_p]
    all_plaid_fish = [p[0][0] for p in all_plaid]
    all_plaid_bouts = [p[1][0] for p in all_plaid]
    all_replay = [a.get() for a in ar_r]
    all_repl_fish = [r[0][0] for r in all_replay]
    all_repl_bouts = [r[1][0] for r in all_replay]

    # global fit parameters
    history = 25  # length of history on temperature and bout ends to consider as inputs
    num_epochs = 10  # the number of training epochs in each round for boutprob, feature is 10x that
    batch_size = 256  # the training batch size
    n_reps = 50  # the number of different fits to create and analyze

    # compute normalization across all fish and conditions
    standardization = utility.InputDataStandard.from_experiment_data(all_plaid_fish + all_repl_fish,
                                                                     all_plaid_bouts + all_repl_bouts, 9)
    # compute average input across all fish and conditions for storage - this avoids having to reload experiments in
    # any analysis file that uses the fit models
    all_plaid_data = utility.Data_BoutProbability(history, all_plaid_fish, all_plaid_bouts, (9,), 1.0, standardization)
    all_repl_data = utility.Data_BoutProbability(history, all_repl_fish, all_repl_bouts, (9,), 1.0, standardization)
    all_dynamic = []
    all_static = []
    train_set = all_plaid_data.training_data(batch_size)
    for inp_dyn, inp_stat, outp in train_set:
        all_dynamic.append(inp_dyn.numpy())
        all_static.append(inp_stat.numpy())
    train_set = all_repl_data.training_data(batch_size)
    for inp_dyn, inp_stat, outp in train_set:
        all_dynamic.append(inp_dyn.numpy())
        all_static.append(inp_stat.numpy())
    x_bar_dyn_pf = np.mean(np.vstack(all_dynamic), 0, keepdims=True)
    x_bar_stat_pf = np.mean(np.vstack(all_static), 0, keepdims=True)

    # create storage file and type subgroups
    with h5py.File(path.join(fit_dir, "fit_and_test_data.hdf5"), 'w') as data_file:
        # store this run's average data
        grp_x_bar = data_file.create_group("data_averages_prob")
        grp_x_bar.create_dataset("x_bar_dyn", data=x_bar_dyn_pf)
        grp_x_bar.create_dataset("x_bar_stat", data=x_bar_stat_pf)

        # store this run's input data standardization
        grp_standard = data_file.create_group("standardization")
        standardization.save_to_hdf5(grp_standard)

        # create groups for experimental types
        grp_naive = data_file.create_group("naive")
        grp_plaid = data_file.create_group("plaid")
        grp_replay = data_file.create_group("replay")
        grp_plaidcontrol = data_file.create_group("plaid_control")

        assert len(all_plaid_fish) == len(all_repl_fish)

        # run train test loop
        for i in range(n_reps):

            # Get training done on our pool
            ar_plaid = train_pool.apply_async(train_test, [all_plaid_fish, all_plaid_bouts, 0.8, history,
                                                           batch_size, standardization, num_epochs, False])
            ar_repl = train_pool.apply_async(train_test, [all_repl_fish, all_repl_bouts, 0.8, history,
                                                          batch_size, standardization, num_epochs, False])
            ar_pcont = train_pool.apply_async(train_test, [all_plaid_fish, all_plaid_bouts, 0.8,
                                                           history, batch_size, standardization, num_epochs, True])
            # Save results to our file
            save_pool_results(ar_plaid, i, grp_plaid, grp_naive)
            save_pool_results(ar_repl, i, grp_replay, None)
            save_pool_results(ar_pcont, i, grp_plaidcontrol, None)

            print()
            print(f"{np.round((i+1)/n_reps*100, 2)} % completed")
            print()
