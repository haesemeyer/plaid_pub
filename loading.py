"""
Utility functions to load relevant experimental data recursively from a root folder
"""

import h5py
import numpy as np
from typing import List, Tuple
from os import path
import os
import pandas as pd


def find_all_exp_paths(root_f: str) -> List[str]:
    """
    Recursively searches for experiments, identified by the presence of .info files
    :param root_f: The folder from which to start the current search
    :return: List of all .info files at and below root_f with their full path
    """
    try:
        objects = os.listdir(root_f)
    except PermissionError:
        return []
    exp_local = [path.join(root_f, o) for o in objects if ".info" in o]
    dir_local = [path.join(root_f, o) for o in objects if path.isdir(path.join(root_f, o))]
    exp_deep = []
    for dl in dir_local:
        exp_deep += find_all_exp_paths(dl)
    return exp_local + exp_deep


def compute_tail_angles(tail_components: np.ndarray) -> np.ndarray:
    """
    Computes tail angles from decomposed tail angles
    :param tail_components: n_frames x n_tailpoints x 2 array of cos/sin decomposed tail angles relative to heading
    :return: n_frames x n_tailpoints array of tail-angles across time
    """
    return np.arctan2(tail_components[:, :, 1], tail_components[:, :, 0])


def load_at_index(indices: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Uses an array of indices to assign data at specific locations
    :param indices: n_frames long array of actual frame indices
    :param data: n_frames long array of data acquired at those frame indices
    :return: max(indices+1) long array where data has been inserted based on a 0-based frame index
    """
    out = np.full(int(np.nanmax(indices)+1), np.nan)
    valid = np.isfinite(indices)
    ix_val = indices[valid].astype(int)
    data_val = data[valid]
    out[ix_val] = data_val
    return out


def load_exp_data_by_info(info_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and combines all relevant experimental data from .track and _MPB.h5_converted.hdf5 files
    :param info_file_path: Full path to the experiments info file
    :return:
        [0]: Pandas dataframe of per-frame fish information
        [1]: Pandas dataframe of per-frame full tail angle information
    """
    if_path, if_file = path.split(info_file_path)
    base_name = if_file[:if_file.find(".info")]
    track_file = path.join(if_path, f"{base_name}.track")
    mpb_file = path.join(if_path, f"{base_name}_MPB.h5_converted.hdf5")
    track_data = np.genfromtxt(track_file, delimiter="\t")
    indices = track_data[:, 0]
    original_frame = load_at_index(indices, np.arange(indices.size))
    laser_power = load_at_index(indices, track_data[:, -1])
    trial_number = load_at_index(indices, track_data[:, -2])
    exp_phase = load_at_index(indices, track_data[:, -3])
    # consider first 5 minutes (5*60*250 frames) of plaid/replay phase as burn-in
    ix_stim_phase = np.where(exp_phase == 9)[0]
    exp_phase[ix_stim_phase[:5*60*250]] = 20
    x_position = load_at_index(indices, track_data[:, 1])
    y_position = load_at_index(indices, track_data[:, 2])
    df_track = pd.DataFrame(np.c_[original_frame, x_position, y_position, laser_power, exp_phase, trial_number],
                            columns=["TFile frame", "X Position", "Y Position",  "Laser Power", "Experiment phase",
                                     "Current trial"])
    with h5py.File(mpb_file, 'r') as dfile:
        # create tail information dataframe
        tail_components = dfile["tailComponents"][()]
        tail_angles = compute_tail_angles(tail_components)
        tp_names = [f"Segment {tail_angles.shape[1]-i-1}" for i in range(tail_angles.shape[1] - 1)] + ["SB"]
        df_tail = pd.DataFrame(tail_angles, columns=tp_names)
        # create part 2 of fish information dataframe
        heading = dfile["smoothedHeading"][()]
        tail_tip_angle = tail_angles[:, 0]
        df_mpb = pd.DataFrame(np.c_[heading, tail_tip_angle], columns=["Heading", "Tail tip angle"])
    assert df_track.shape[0] == df_mpb.shape[0] == df_tail.shape[0], info_file_path
    return pd.concat((df_track, df_mpb), axis=1), df_tail


def load_track_data_by_info(info_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and combines all relevant experimental data from .track and _MPB.h5_converted.hdf5 files
    for pure Track and Write experiments
    :param info_file_path: Full path to the experiments info file
    :return:
        [0]: Pandas dataframe of per-frame fish information
        [1]: Pandas dataframe of per-frame full tail angle information
    """
    if_path, if_file = path.split(info_file_path)
    base_name = if_file[:if_file.find(".info")]
    track_file = path.join(if_path, f"{base_name}.track")
    mpb_file = path.join(if_path, f"{base_name}_MPB.h5_converted.hdf5")
    track_data = np.genfromtxt(track_file, delimiter="\t")
    indices = track_data[:, 0]
    original_frame = load_at_index(indices, np.arange(indices.size))
    x_position = load_at_index(indices, track_data[:, 1])
    y_position = load_at_index(indices, track_data[:, 2])
    df_track = pd.DataFrame(np.c_[original_frame, x_position, y_position],
                            columns=["TFile frame", "X Position", "Y Position"])
    with h5py.File(mpb_file, 'r') as dfile:
        # create tail information dataframe
        tail_components = dfile["tailComponents"][()]
        tail_angles = compute_tail_angles(tail_components)
        tp_names = [f"Segment {tail_angles.shape[1]-i-1}" for i in range(tail_angles.shape[1] - 1)] + ["SB"]
        df_tail = pd.DataFrame(tail_angles, columns=tp_names)
        # create part 2 of fish information dataframe
        heading = dfile["smoothedHeading"][()]
        tail_tip_angle = tail_angles[:, 0]
        df_mpb = pd.DataFrame(np.c_[heading, tail_tip_angle], columns=["Heading", "Tail tip angle"])
    assert df_track.shape[0] == df_mpb.shape[0] == df_tail.shape[0]
    return pd.concat((df_track, df_mpb), axis=1), df_tail


if __name__ == '__main__':
    pass
