import h5py
import numpy as np
import argparse
from typing import Any, List
from os import path
import os
import pandas as pd


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'folder':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        elif self.dest == 'processes':
            if values <= 0:
                raise argparse.ArgumentError(self, "The number of processes to use has to be larger than 0.")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def find_all_mpb_paths(root_f: str) -> List[str]:
    """
    Recursively searches for MPB files
    :param root_f: The folder from which to start the current search
    :return: List of all MPB.h5 files at and below root_f
    """
    try:
        objects = os.listdir(root_f)
    except PermissionError:
        return []
    mpb_local = [path.join(root_f, o) for o in objects if "MPB.h5" in o and "_converted" not in o]
    dir_local = [path.join(root_f, o) for o in objects if path.isdir(path.join(root_f, o))]
    mpb_deep = []
    for dl in dir_local:
        mpb_deep += find_all_mpb_paths(dl)
    return mpb_local + mpb_deep


def convert_mpb_file(data_file_path: str, redo: bool) -> str:
    """
    Convert mpb file saving analysis relevant information in new hdf5 file
    :param data_file_path: Path to the file to be converted
    :param redo: If set to false file won't be converted again if a converted file is already present
    :return: Status message
    """
    f_path, f_name = path.split(data_file_path)
    new_name = f_name + "_converted.hdf5"
    if not redo and path.exists(path.join(f_path, new_name)):
        return "Skipped conversion. File exists."
    # first read dataframes (Jamie data and raw data)
    flhc = pd.read_hdf(data_file_path, key="flhc")
    te_position = np.vstack(flhc['mideye'])
    sb_position = np.vstack(flhc['bladder'])
    raw = pd.read_hdf(data_file_path, key="rawdata")
    x_raw = np.hstack(raw['x'])  # the save-frame center x coordinate
    y_raw = np.hstack(raw['y'])  # the save-frame center y coordinate
    c_raw = np.c_[x_raw, y_raw]
    # convert from save-frame to world coordinates
    te_position += c_raw
    sb_position += c_raw
    with h5py.File(data_file_path, 'r') as mpb_file:
        sm_heading = mpb_file["smoothedheading"][()][:, None]
        sm_tail_angles = mpb_file["smoothedtailangles"][()]
        # perform heading correction of tail angles
        heading_corr_tail_angles = sm_tail_angles - sm_heading
        # compute vector components of corrected tail angles
        tail_components = np.full(sm_tail_angles.shape + (2,), np.nan)
        tail_components[:, :, 0] = np.cos(heading_corr_tail_angles)
        tail_components[:, :, 1] = np.sin(heading_corr_tail_angles)
    if sm_heading.shape[0] != tail_components.shape[0] or sm_heading.shape[0] != te_position.shape[0] or \
            sm_heading.shape[0] != sb_position.shape[0]:
        print()
        print("!!!!!!!!!!")
        print(f"Mismatch in frame counts in file {data_file_path}. Skip saving data.")
        print("!!!!!!!!!!")
        print()
        return "Skipped conversion. Error in file data."
    with h5py.File(path.join(f_path, new_name), 'w') as conv_file:
        conv_file.create_dataset("tailComponents", data=tail_components, compression="gzip", compression_opts=9)
        conv_file.create_dataset("smoothedHeading", data=sm_heading, compression="gzip", compression_opts=9)
        conv_file.create_dataset("sbPosition", data=sb_position, compression="gzip", compression_opts=9)
        conv_file.create_dataset("tePosition", data=te_position, compression="gzip", compression_opts=9)
    return "Converted."


if __name__ == '__main__':
    """
    The following data will be stored in the converted MPB files:

    tailComponents: <nframes x 20 tail-segments x 2> array of cosine/sine decomposed heading-corrected
        tail angles.
        Example:
            [20000, 0, 0] indexes out the cosine of the tail tip angle in frame 20000
            [20000, -1, 1] indexes out the sine of the swim bladder angle in frame 20000

    smoothedHeading: <nframes> long vector of median filtered heading angles (same as in JC file)

    sbPosition: <nframes x 2> matrix of swim bladder x,y world coordinates (the fish position)

    tePosition: <nframes x 2> matrix of third eye x,y world coordinates (the head center position)
    """

    a_parser = argparse.ArgumentParser(prog="convert_jc_data",
                                       description="Extracts and stores data relevant for prob-comp model from Jamie"
                                                   "Costabile's tail extraction experiment files")
    a_parser.add_argument("-f", "--folder", help="Path to folder that will be recursively searched for MPB.h5 files",
                          type=str, default="", action=CheckArgs)
    a_parser.add_argument("-r", "--redo", help="If set conversion will be redone", action='store_true')

    args = a_parser.parse_args()

    data_folder = args.folder
    rd = args.redo

    all_MPB_files = find_all_mpb_paths(data_folder)

    for f in all_MPB_files:
        print(f"Processing {path.split(f)[0]}.")
        status = convert_mpb_file(f, rd)
        print(f"{status}")
