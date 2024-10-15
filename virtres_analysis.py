"""
Script to see whether activity in mixed selectivity neurons (that are nonlinear and respond to stimulus as well as
bout starts) can be used to classify experiments into Plaid and Replay categories
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
import h5py
from typing import List, Tuple
import copy
import random
import re
from os import path
from sys import exit
import os
import matplotlib.pyplot as pl
import matplotlib as mpl
import seaborn as sns


def reduce_dim_cluster(neuro_data: np.ndarray, ndim: int) -> np.ndarray:
    """
    Reduce dimensionality of input data via clustering
    :param neuro_data: n_neurons x t_timepoints matrix of neural response data
    :param ndim: The desired number of dimensions
    :return: n_dim x t_timepoints matrix of reduced dimensionality data
    """
    # calculate pairwise correlations
    corr_mat = np.corrcoef(neuro_data)
    corr_mat[np.isnan(corr_mat)] = 0
    corr_mat[corr_mat < 0.2] = 0
    spc = SpectralClustering(n_clusters=ndim, affinity='precomputed')
    spc.fit(corr_mat)
    reduced = np.empty((ndim, neuro_data.shape[1]))
    for i in range(ndim):
        reduced[i, :] = np.mean(neuro_data[spc.labels_ == i], 0)
    return reduced


def reduce_dim_pca(neuro_data: np.ndarray, ndim: int) -> np.ndarray:
    """
    Reduce dimensionality of input data via PCA
    :param neuro_data: n_neurons x t_timepoints matrix of neural response data
    :param ndim: The desired number of dimensions
    :return: n_dim x t_timepoints matrix of reduced dimensionality data
    """
    pca = PCA(n_components=ndim)
    pca.fit(neuro_data.T)  # need to transpose, since PCA expects n_features x n_samples
    print(f"PCA across {ndim} components explains {np.round(np.sum(pca.explained_variance_ratio_)*100)} "
          f"% of total variance.")
    return pca.transform(neuro_data.T).T


def create_matched_tuples(neuro_data: np.ndarray, exp_id: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Based on concatenated neural data and experiment ids generates list of matched Plaid Replay pairs
    :param neuro_data: The concatenated (dim-reduced) neural data
    :param exp_id: Vector of experiment ids
    :return: List of matched Plaid-Replay experiment pairs
    """
    exp_count = (np.max(exp_id) - 1) // 10
    assert exp_count == 52  # current number
    paired_list = []
    for i in range(1, exp_count+1):  # these numbering indices are 1-based
        plaid = neuro_data[:, exp_id == i*10].copy()
        replay = neuro_data[:, exp_id == i*10+1].copy()
        paired_list.append((plaid, replay))
    return paired_list


def generate_input_samples(neuro_data: np.ndarray, n_consecutive: int) -> np.ndarray:
    """
    Generate design matrix with variable number of consecutive samples
    :param neuro_data: n_dim x t_timepoints matrix of reduced neural response data
    :param n_consecutive: The number of consecutive timepoints to use as inputs
    :return: (n_dim * n_consecutive) * (t_timepoints // n_consecutive) sized design matrix (features x samples)
    """
    x = np.empty((neuro_data.shape[0]*n_consecutive, neuro_data.shape[1] - n_consecutive))
    for i in range(x.shape[1]):
        x[:, i] = neuro_data[:, i:i+n_consecutive].ravel()
    return x


def create_input_output_samples(paired_list: List[Tuple[np.ndarray, np.ndarray]],
                                n_consecutive: int, n_thin=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a set of experiments creates design matrix and outputs for classifier (NOTE: This function inverts from
    features x samples to samples x features to make the output compatible with sklearn models
    :param paired_list: List of Plaid, Replay tuples
    :param n_consecutive: The number of consecutive timepoints to include in modeling
    :param n_thin: Thinning of samples, only returning every n-th sample
    :return:
        [0]: (n_timepoints//n_consecutive x len(paired_list)) x (n_consecutive x n_dim) design matrix (samplesxfeatures)
        [1]: (n_timepoints//n_consecutive x len(paired_list)) of 0(Plaid) / 1(Replay) classifier outputs
    """
    ins, outs = [], []
    for pair in paired_list:
        x = generate_input_samples(pair[0], n_consecutive).T
        y = np.zeros(x.shape[0])
        ins.append(x)
        outs.append(y)
        x = generate_input_samples(pair[1], n_consecutive).T
        y = np.ones(x.shape[0])
        ins.append(x)
        outs.append(y)
    return np.vstack(ins)[::n_thin, :], np.hstack(outs)[::n_thin, None]


def spectral_cluster(in_data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Performs spectral clustering with subsequent correlation based tightening on the data
    :param in_data: n_samples x n_features matrix of input data
    :param n_clusters: The number of clusters to form
    :return: n_samples long vector of cluster memberships (-1 for not clustered)
    """
    membership = np.full(in_data.shape[0], -1, dtype=int)
    # create pairwise correlations matrix
    corr_mat = np.corrcoef(in_data)
    corr_mat[np.isnan(corr_mat)] = 0
    corr_mat[corr_mat < 0.2] = 0
    spc = SpectralClustering(n_clusters, affinity="precomputed")
    spc.fit(corr_mat)
    initial_labels = spc.labels_
    # extract cluster averages for angle-based refinement
    cluster_avgs = np.zeros((n_clusters, in_data.shape[1]))
    for i in range(n_clusters):
        if np.sum(initial_labels == i) > 1:
            cluster_avgs[i, :] = np.mean(in_data[initial_labels == i, :], 0)
        else:
            cluster_avgs[i, :] = in_data[initial_labels == i, :]
    # calculate correlation of each trace to each cluster centroid and assign to either max correlation or -1
    for i, tr in enumerate(in_data):
        cl_num = -1
        c_max = -1
        for j in range(n_clusters):
            c = np.corrcoef(tr, cluster_avgs[j])[0, 1]
            if c > 0.6 and c > c_max:
                cl_num = j
                c_max = c
        membership[i] = cl_num
    return membership


if __name__ == '__main__':
    mpl.rcParams['pdf.fonttype'] = 42
    a_parser = argparse.ArgumentParser(prog="neuron_class",
                                       description="Uses mixed selectivity neurons to classify Plaid vs. Replay")
    a_parser.add_argument("-f", "--file", help="Path to hdf5 file with neural data", type=str, required=True)
    a_parser.add_argument("-dm", "--dim", help="Dimensionality reduction method [cluster, pca]", type=str,
                          required=True)
    a_parser.add_argument("-n", "--ndim", help="Number of dimensions to retain", type=int, default=10)

    args = a_parser.parse_args()
    dfilename = args.file
    if not path.exists(dfilename) or not path.isfile(dfilename):
        print(f"{dfilename} is not a valid file")
        exit()
    dim_method = str.lower(args.dim)
    if dim_method != "cluster" and dim_method != "pca":
        print(f"Dimensionality reduction method has to be one of <cluster, pca>. Not <{dim_method}>")
        exit()
    nd = args.ndim

    plot_dir = "virtual_response_analysis"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    hyper_grid = {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}

    # Load neural data from hdf5 file, concatenate and create experiment ID vector
    neurons = []
    exp_ids = []
    stim_plus_starts = []
    with h5py.File(dfilename, 'r') as dfile:
        for k in dfile:
            if "Plaid" not in k and "Replay" not in k:
                # This is not a group with Plaid or replay data
                continue
            # Use regular expressions to identify the experiment number
            num_cand = re.search(r"_\d+", k)
            fish_number = int(num_cand.group()[1:]) + 1
            eid = (fish_number*10 + 1) if "Replay" in k else (fish_number*10)
            # load stimulus and bout-starts as comparison predictors
            stim = dfile[k]["Stimulus"][()][None, 49:]
            starts = dfile[k]["Starts"][()][None, 49:]
            dangle = dfile[k]["DAngle"][()][None, 49:]
            disp = dfile[k]["Displacement"][()][None, 49:]
            stim_plus_starts.append(np.r_[stim, starts, dangle, disp])
            # load neural respones
            grp = dfile[k]
            neuron_grp = grp["Neuron responses"]
            n_neurons = len(neuron_grp.keys())
            these_neurons = []  # after vstacking: n_neurons x n_timepoints matrix
            for nid in neuron_grp:
                these_neurons.append(neuron_grp[nid][()])
            these_neurons = np.vstack(these_neurons)
            neurons.append(these_neurons)
            exp_ids.append(np.ones(these_neurons.shape[1], dtype=int) * eid)

    neurons = np.hstack(neurons)  # hstack since we want to stack experiments in sequence
    stim_plus_starts = np.hstack(stim_plus_starts)  # same
    exp_ids = np.hstack(exp_ids)

    n_iterations = 100
    use_consecutive = [1, 5, 10, 25, 50]
    if dim_method == "cluster":
        neurons = reduce_dim_cluster(neurons, nd)
    elif dim_method == "pca":
        neurons = reduce_dim_pca(neurons, nd)
    else:
        print("Unknown dimensionality reduction method")
        exit()

    exp_pairs = create_matched_tuples(neurons, exp_ids)
    exp_pairs_original = copy.deepcopy(exp_pairs)  # store for later reference since other list will be shuffled

    n_pairs = len(exp_pairs)
    n_train = 2*n_pairs // 3
    thin = 5

    auc_dict = {"Classifier": [], "Consecutive": [], "AUC": []}

    cons = 1
    print()
    print(f"Dimensionality reduction method: {dim_method}")
    print(f"Number of dimensions: {nd}")
    print()
    for iteration in range(n_iterations):
        # create random split into train and test pairs
        random.shuffle(exp_pairs)
        train_set = exp_pairs[:n_train]
        test_set = exp_pairs[n_train:]

        x_train, y_train = create_input_output_samples(train_set, cons, thin)
        x_test, y_test = create_input_output_samples(test_set, cons, thin)

        m = np.mean(np.r_[x_train, x_test], axis=0, keepdims=True)
        s = np.std(np.r_[x_train, x_test], axis=0, keepdims=True)
        x_train -= m
        x_train /= s
        x_test -= m
        x_test /= s

        class_model = LogisticRegressionCV(penalty="l2", Cs=hyper_grid['C'], solver='saga', max_iter=250, n_jobs=10)
        class_model.fit(x_train, y_train.ravel())
        auc = roc_auc_score(y_test, class_model.predict_proba(x_test)[:, 1])
        auc_dict["Consecutive"].append(cons)
        auc_dict["AUC"].append(auc)
        auc_dict["Classifier"].append("Logistic regression")
        print()
        print(f"For {cons} consecutive samples LR ROC AUC is {auc}.")

    # comparison analysis, directly using stimulus and behavior for the same prediction
    spb_pairs = create_matched_tuples(stim_plus_starts, exp_ids)
    for cons in use_consecutive:
        print()
        print(f"Direct stimulus and behavior comparison")
        print()
        for iteration in range(n_iterations):
            # create random split into train and test pairs
            random.shuffle(spb_pairs)
            train_set = spb_pairs[:n_train]
            test_set = spb_pairs[n_train:]

            x_train, y_train = create_input_output_samples(train_set, cons, thin)
            x_test, y_test = create_input_output_samples(test_set, cons, thin)

            m = np.mean(np.r_[x_train, x_test], axis=0, keepdims=True)
            s = np.std(np.r_[x_train, x_test], axis=0, keepdims=True)
            x_train -= m
            x_train /= s
            x_test -= m
            x_test /= s

            class_model = LogisticRegressionCV(penalty="l2", Cs=hyper_grid['C'], solver='saga', max_iter=250, n_jobs=10)
            class_model.fit(x_train, y_train.ravel())
            auc = roc_auc_score(y_test, class_model.predict_proba(x_test)[:, 1])
            auc_dict["Consecutive"].append(cons)
            auc_dict["AUC"].append(auc)
            auc_dict["Classifier"].append("Input LR")
            print()
            print(f"For {cons} consecutive samples Input LR ROC AUC is {auc}.")

    with h5py.File(path.join(plot_dir, "Neuron_Class_Results.hdf5"), 'w') as dfile:
        # Save AUC dictionary to file
        for k in auc_dict:
            dfile.create_dataset(k, data=auc_dict[k])

    df_auc = pd.DataFrame(auc_dict)
    fig = pl.figure(figsize=(12, 4.8))
    sns.boxplot(data=df_auc, x="Consecutive", y="AUC", whis=np.Inf,
                hue="Classifier")
    sns.stripplot(data=df_auc, x="Consecutive", y="AUC",
                  color=[0.8, 0.8, 0.8], hue="Classifier", dodge=True)
    pl.ylim(0.4, 0.7)
    pl.yticks([0.4, 0.5, 0.6, 0.7])
    sns.despine()
    fig.savefig(path.join(plot_dir, "NeuronClassifier_Performance.pdf"))

    # Overview plot of stimulus/behavior in context of neural activity
    example_neurons = []
    with h5py.File(dfilename, 'r') as dfile:
        grp = dfile["Plaid_0"]
        bout_frames = np.where(grp["Starts"][1250:1550] > 0)[0]
        stimulus = grp["Stimulus"][1250:1550]
        neuron_grp = grp["Neuron responses"]
        for k in neuron_grp:
            example_neurons.append(neuron_grp[k][()])
    example_neurons = np.vstack(example_neurons)[:, 1250:1550]

    time = np.arange(stimulus.size) / 5
    fig = pl.figure()
    pl.plot(time, stimulus)
    pl.vlines(time[bout_frames], 30, 32, color='k')
    pl.xlabel("Time [s]")
    fig.savefig(path.join(plot_dir, "Example_Stim_Swim.pdf"))

    cluster_numbers = spectral_cluster(example_neurons, 20)
    val_neurons = example_neurons[cluster_numbers > -1]
    val_numbers = cluster_numbers[cluster_numbers > -1]
    fig = pl.figure()
    sns.heatmap(val_neurons[np.argsort(val_numbers)], yticklabels=250, cmap='bone', rasterized=True, vmin=-4, vmax=4)
    fig.savefig(path.join(plot_dir, "Example_Activity.pdf"), dpi=600)
