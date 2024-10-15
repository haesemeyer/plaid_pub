"""
Script to analyze model performance on train vs. test sets with increasing numbers of training epochs
"""

from analysis import CheckArgs
from fit_models import load_all
import argparse
import loading
import matplotlib as mpl
from os import path
import os
import matplotlib.pyplot as pl
import seaborn as sns
import utility
import trainers
from sklearn.metrics import roc_auc_score
import multiprocessing as mp


if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="hyper_param.py",
                                       description="Uses separate dataset to estimate best l2 penalty for models")
    a_parser.add_argument("-pf", "--p_folder", help="Path to folder with Plaid experiments", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-rf", "--r_folder", help="Path to folder with Replay experiments", type=str, default="",
                          action=CheckArgs)

    args = a_parser.parse_args()

    plaid_folder = args.p_folder
    replay_folder = args.r_folder

    plaid_exps = loading.find_all_exp_paths(plaid_folder)
    replay_exps = loading.find_all_exp_paths(replay_folder)

    plot_dir = "hyper_plots_l2"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # load and process all data
    load_pool = mp.Pool(5)
    ar_p = [load_pool.apply_async(load_all, [[pe]]) for pe in plaid_exps]
    ar_r = [load_pool.apply_async(load_all, [[pr]]) for pr in replay_exps]
    all_plaid = [a.get() for a in ar_p]
    all_plaid_fish = [p[0][0] for p in all_plaid]
    all_plaid_bouts = [p[1][0] for p in all_plaid]
    all_replay = [a.get() for a in ar_r]
    all_repl_fish = [r[0][0] for r in all_replay]
    all_repl_bouts = [r[1][0] for r in all_replay]

    all_fish = all_plaid_fish + all_repl_fish
    all_bouts = all_plaid_bouts + all_repl_bouts

    # global fit parameters
    history = 25  # length of history on temperature and bout ends to consider as inputs
    num_epochs = 10  # the number of training epochs in each round for boutprob, feature is 10x that
    batch_size = 256  # the training batch size
    n_reps = 5  # the number of different fits to create and analyze

    l2_to_test = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    # compute standardization across all experiments
    standardization = utility.InputDataStandard.from_experiment_data(all_fish, all_bouts, 9)

    prob_test = {"l2 penalty": [], "AUC": []}

    for l2_tt in l2_to_test:
        for repeat in range(n_reps):
            print(f"Beginning repeat {repeat} for l2 penalty {l2_tt}")
            # generate model trainers for this repeat
            prob_trainer = trainers.BoutProbabilityTrainer(history, 1024, standardization, all_fish, all_bouts, 0.8,
                                                           False, l2_tt)
            # train models
            prob_trainer.train(num_epochs)
            # get test data from trainers
            prob_true, prob_pred = prob_trainer.test()
            # compute test metrics - AUC for probability qq-slope for distance and magnitude
            prob_auc = roc_auc_score(prob_true > 0.5, prob_pred)
            prob_test["l2 penalty"].append(l2_tt)
            prob_test["AUC"].append(prob_auc)

            print(f"Test scores for l2 penalty {l2_tt}")
            print(f"Prob AUC: {prob_auc}")
            print()

    fig = pl.figure()
    sns.boxplot(data=prob_test, x="l2 penalty", y="AUC")
    sns.stripplot(data=prob_test, x="l2 penalty", y="AUC", color='k')
    sns.despine()
    fig.savefig(path.join(plot_dir, "prob_auc.pdf"))
