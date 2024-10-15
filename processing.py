"""
Utility functions for behavioral data processing
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numba

import utility


def average_angle(angles: np.ndarray) -> np.ndarray:
    """
    Computes an average angle through decomposition
    :param angles: The individual angles to average in radians
    :return: The average angle
    """
    x, y = np.cos(angles), np.sin(angles)
    return np.arctan2(np.mean(y), np.mean(x))


def arc_distance(angles: np.ndarray) -> np.ndarray:
    """
    Computes the length of the arcs between consecutive angles on the unit circle
    :param angles: n_frames long vector of the angles in radians between which to compute the arc distances
    :return: n_frames-1 long vector of arc distances
    """
    x, y = np.cos(angles), np.sin(angles)
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    theta = np.arccos(1 - d**2 / 2)
    return theta  # since on the unit circle the arc-distance is equal to the angular difference


def fix_angle_trace(angles: np.ndarray, ad_thresh: float, max_ahead: int) -> np.ndarray:
    """
    Tries to fix sudden jumps in angle traces (angle differences that are implausible) by filling stretches between
    with the pre-jump angle
    :param angles: The angle trace to fix
    :param ad_thresh: The largest plausible delta-angle (arc distance on unit circle, i.e. smallest angular difference)
    :param max_ahead: The maximal amount of frames to look ahead for a jump back (longer stretches won't be fixed)
    :return: The corrected angle trace
    """
    adists = np.r_[0, arc_distance(angles)]
    angles_corr = np.full(angles.size, np.nan)
    index = 0
    while index < adists.size:
        ad = adists[index]
        if np.isnan(ad):
            angles_corr[index] = angles[index]
            index += 1
            continue
        if ad < ad_thresh:
            angles_corr[index] = angles[index]
            index += 1
        else:
            # start correction loop
            next_jump_ix = index + 1
            for i in range(max_ahead):
                if i + next_jump_ix >= adists.size:
                    # nothing we can do here, just set this one to NaN by not filling it and continue
                    break
                if adists[i + next_jump_ix] >= ad_thresh:
                    # we found a similar jump within the next ten frames fill with initial angle to correct
                    replace_angle = angles_corr[index - 1]
                    assert np.isfinite(replace_angle)
                    angles_corr[index:i + next_jump_ix + 1] = replace_angle
                    index = next_jump_ix + i
                    assert np.isfinite(angles_corr[index])
                    break
            index += 1
    return angles_corr


@numba.njit
def vigor(tail_tip_angle, vigor_out, winlen):
    """
    Computes the swim vigor based on a cumulative angle trace
    as the windowed standard deviation of the cumAngles
    """
    s = tail_tip_angle.size
    for i in range(winlen, s):
        angs = tail_tip_angle[i-winlen+1:i+1]
        std = np.nanstd(angs)
        vigor_out[i] = std


@numba.njit
def predict_temperature(lpower: np.ndarray, dt: float) -> np.ndarray:
    """
    Uses temperature model to predict the fish temperature from laser power at sample
    :param lpower: The laser power at sample in W
    :param dt: The timestep of the lpower trace
    :return: The predicted temperatures
    """
    baseline = 24  # The baseline temperature in C
    alpha = 8.8  # The heating rate in K/J
    r = 0.0165  # The cooling rate in 1/frame - derived from step-heating profile after deconv. of thermistor t-const
    temp = np.zeros_like(lpower)
    for t in range(1, temp.size):
        temp[t] = (1-r)*temp[t-1] + dt*alpha*lpower[t]
    return temp + baseline


def pre_process_fish_data(fish_data: pd.DataFrame, px_per_mm=8.0, frame_rate=250) -> None:
    """
    Performs filtering and interpolation on fish data for smoothening and small gap filling and adds
    fish speeds and tail vigor to the data. Converts coordinates to mm and expresses speed in mm/s
    :param fish_data: raw Fish DataFrame returned by load_exp_data_by_info
    :param px_per_mm: The spatial resolution of the acquisition in pixels per mm
    :param frame_rate: The temporal resolution of the acquisition in Hz
    """
    # Filter position data, setting sigma to 1/3 of expected swim bout length at 250 Hz
    # and convert to mm
    fish_data["Raw X"] = fish_data["X Position"].copy()
    fish_data["Raw Y"] = fish_data["Y Position"].copy()
    fish_data["X Position"] = gaussian_filter1d(fish_data["X Position"], sigma=7) / px_per_mm
    fish_data["Y Position"] = gaussian_filter1d(fish_data["Y Position"], sigma=7) / px_per_mm
    # Compute instantaneous speed
    i_speed = np.sqrt(np.diff(fish_data["X Position"])**2 + np.diff(fish_data["Y Position"])**2)
    # NOTE: For time conversion we should maybe convert frame numbers int times to account for dropped frames
    i_speed = np.r_[0, i_speed] * frame_rate
    fish_data["Instant speed"] = i_speed
    # filter out improbably heading angle jumps - 1 radians was determined as the cut-off
    # through an arc_distance histogram
    fish_data["Heading"] = fix_angle_trace(fish_data["Heading"], 1, 10)
    # compute tail vigor
    tta = np.array(fish_data["Tail tip angle"])
    vig = np.ones(tta.size)
    vigor(tta, vig, 10)
    fish_data["Tail vigor"] = vig
    if "Laser Power" in fish_data:
        l_power = np.array(fish_data["Laser Power"])
        l_power = utility.mean_fill_nan(l_power)
        fish_data["Temperature"] = predict_temperature(l_power/1000, 1/frame_rate)


def identify_bouts(fish_data: pd.DataFrame, vig_thresh, frame_rate=250) -> pd.DataFrame:
    """
    Uses tail vigor to identify swim bouts and returns dataframe with bout characteristics
    :param fish_data: Fish data after pre-processing
    :param vig_thresh: Vigor threshold for calling bouts
    :param frame_rate: The temporal resolution of the acquisition in Hz
    :return: Dataframe with bout data (start, stop, peak speed, displacement, distance, angle-change)
    """
    vig_trace = np.array(fish_data["Tail vigor"])
    above_th = (vig_trace > vig_thresh).astype(float)
    starts_stops = np.r_[0, np.diff(above_th)]
    start_frames = np.where(starts_stops > 0)[0]
    stop_frames = np.where(starts_stops < 0)[0]
    if stop_frames.size > start_frames.size or stop_frames[0] < start_frames[0]:
        stop_frames = stop_frames[1:]  # this can only occur if the vigor is high to begin with which should not happen
    if start_frames.size > stop_frames.size:
        start_frames = start_frames[:stop_frames.size]
    bout_lengths = stop_frames - start_frames + 1
    valid = np.logical_and(bout_lengths >= 20, bout_lengths <= 75)  # at least 80 ms maximally 300 ms
    valid = np.logical_and(valid, stop_frames < vig_trace.size-1)  # if a bout ends right at experiment end remove
    start_frames = start_frames[valid]
    stop_frames = stop_frames[valid]
    if start_frames.size == 0:
        return pd.DataFrame(columns=["Start", "Stop", "Peak speed", "Displacement", "Angle change",
                                     "Experiment phase", "Average vigor", "IBI", "Maintain"])
    p_speeds = np.full(start_frames.size, np.nan)
    displace = p_speeds.copy()
    angle_change = p_speeds.copy()
    phase = p_speeds.copy()
    avg_vigor = p_speeds.copy()
    ibi = p_speeds.copy()
    for i, (s, e) in enumerate(zip(start_frames, stop_frames)):
        try:
            phase[i] = fish_data["Experiment phase"][s]
        except KeyError:
            pass
        p_speeds[i] = np.max(fish_data["Instant speed"][s:e+1])
        pre_start = s-5 if s >= 5 else 0
        post_end = e+5 if (e+5) < vig_trace.size else vig_trace.size-1
        pre_slice = slice(pre_start, s)
        post_slice = slice(e+1, post_end)
        pre_x = np.mean(fish_data["X Position"][pre_slice])
        pre_y = np.mean(fish_data["Y Position"][pre_slice])
        post_x = np.mean(fish_data["X Position"][post_slice])
        post_y = np.mean(fish_data["Y Position"][post_slice])
        displace[i] = np.sqrt((post_x - pre_x)**2 + (post_y - pre_y)**2)
        pre_angle = average_angle(fish_data["Heading"][pre_slice])
        post_angle = average_angle(fish_data["Heading"][post_slice])
        d_angle = post_angle - pre_angle
        if d_angle > np.pi:
            angle_change[i] = d_angle - 2*np.pi
        elif d_angle < -np.pi:
            angle_change[i] = d_angle + 2*np.pi
        else:
            angle_change[i] = d_angle
        avg_vigor[i] = np.mean(vig_trace[s:e+1])
    # interbout intervals in ms
    ibi[1:] = (start_frames[1:] - stop_frames[:-1]) / frame_rate * 1000
    # limit bouts to those that have no NaN within their frames
    val_bouts = np.logical_and(np.isfinite(p_speeds), np.isfinite(displace))
    val_bouts = np.logical_and(val_bouts, np.isfinite(angle_change))
    flip_maintain = utility.get_flip_maintain(angle_change, 3)
    return pd.DataFrame(np.c_[start_frames, stop_frames, p_speeds, displace, angle_change, phase, avg_vigor, ibi,
                        flip_maintain], columns=["Start", "Stop", "Peak speed", "Displacement", "Angle change",
                                                 "Experiment phase", "Average vigor", "IBI", "Maintain"])[val_bouts]


if __name__ == '__main__':
    pass
