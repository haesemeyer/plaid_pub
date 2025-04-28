"""
Module with helper functions
"""

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from typing import Union, Optional, Tuple, Any, List
import model_defs
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest, norm


@dataclass
class InputDataStandard:
    """Class for consistent standardization of input data"""
    stimulus_mean: float
    stimulus_std: float
    bout_end_mean: float
    bout_end_std: float
    static_mean: np.ndarray
    static_std: np.ndarray

    def save_to_hdf5(self, file_object: Union[h5py.File, h5py.Group], overwrite=False) -> None:
        """
        Saves all contents to a hdf5 file or group object
        :param file_object: The file/group to save to
        :param overwrite: If true will overvwrite data in the file
        """
        create_overwrite(file_object, "stimulus_mean", self.stimulus_mean, overwrite)
        create_overwrite(file_object, "stimulus_std", self.stimulus_std, overwrite)
        create_overwrite(file_object, "bout_end_mean", self.bout_end_mean, overwrite)
        create_overwrite(file_object, "bout_end_std", self.bout_end_std, overwrite)
        create_overwrite(file_object, "static_mean", self.static_mean, overwrite)
        create_overwrite(file_object, "static_std", self.static_std, overwrite)

    @staticmethod
    def from_hdf5(file_object: Union[h5py.File, h5py.Group]):
        """
        Loads a standardization object from an hdf5 file
        :param file_object: The storage location
        :return: Corresponding InputDataStandard object
        """
        stim_m = file_object["stimulus_mean"][()]
        stim_s = file_object["stimulus_std"][()]
        be_m = file_object["bout_end_mean"][()]
        be_s = file_object["bout_end_std"][()]
        st_m = file_object["static_mean"][()]
        st_s = file_object["static_std"][()]
        return InputDataStandard(stim_m, stim_s, be_m, be_s, st_m, st_s)

    @staticmethod
    def from_experiment_data(experiments: Union[pd.DataFrame, List[pd.DataFrame]],
                             bouts: Union[pd.DataFrame, List[pd.DataFrame]], phase: int):
        all_stimulus = []
        all_bout_ends = []
        all_static_data = []
        for experiment_data, bout_data in zip(experiments, bouts):
            exp_phase = np.array(experiment_data["Experiment phase"])
            stimulus = np.array(experiment_data["Temperature"])
            bout_ends = np.zeros(stimulus.size)
            b_end_frames = np.array((bout_data["Stop"])).astype(int)
            bout_ends[b_end_frames] = 1
            all_stimulus.append(stimulus[exp_phase == phase])
            all_bout_ends.append(bout_ends[exp_phase == phase])
            # create n_frames x 2 matrix of previous bout indicators
            prev_bout = np.zeros((stimulus.size, 2))
            b_disps = np.array(bout_data["Displacement"])
            b_mags = np.abs(np.array(bout_data["Angle change"]))
            for i, stop in enumerate(b_end_frames):
                static_data = np.c_[b_disps[i], b_mags[i]]
                if i == b_end_frames.size - 1:
                    # this is the last one so fill this information until the end
                    prev_bout[stop:, :] = static_data
                else:
                    # fill until the next bout end (inclusive)
                    prev_bout[stop:b_end_frames[i + 1] + 1, :] = static_data
            all_static_data.append(prev_bout[exp_phase == phase, :])
        all_stimulus = np.hstack(all_stimulus)
        all_bout_ends = np.hstack(all_bout_ends)
        all_static_data = np.vstack(all_static_data)
        # bin data as in training data generation before calculating standardization
        all_stimulus = mean_bin_1d(all_stimulus, 10)
        all_bout_ends = max_bin_1d(all_bout_ends, 10)
        all_static_data = all_static_data[::10, :][:all_bout_ends.size, :]
        return InputDataStandard(np.nanmean(all_stimulus), np.nanstd(all_stimulus), np.nanmean(all_bout_ends),
                                 np.nanstd(all_bout_ends), np.nanmean(all_static_data, 0),
                                 np.nanstd(all_static_data, 0))


class Data_BoutProbability:
    def __init__(self, input_steps, experiments: Union[pd.DataFrame, List[pd.DataFrame]],
                 bouts: Union[pd.DataFrame, List[pd.DataFrame]], phases: Tuple[int],
                 train_fraction=1.0, standards: Optional[InputDataStandard] = None, shuffle: bool = False):
        """
        Creates a new Data class
        :param input_steps: The number of  timesteps into the past to use to model the response
        :param experiments: Pandas dataframe (or list of dataframes) of all experimental data at all timepoints
        :param bouts: Pandas dataframe (or list of dataframes) of bout data from the same experiment
        :param phases: Tuple with one or more experiment phases that should be encapsulated in the dataset
        :param train_fraction: The fraction of data (by time not bouts) in each phase to use for training
        :param standards: Global standardization of ANN inputs
        :param shuffle: If set to true, rotate outputs with respect to inputs by 1/3 of the data-length
        """
        if train_fraction < 0 or train_fraction > 1:
            raise ValueError(f"Train fraction has to be between 0 and 1 not {train_fraction}")

        try:
            iter(experiments)
        except TypeError:
            experiments = [experiments]

        try:
            iter(bouts)
        except TypeError:
            bouts = [bouts]

        if len(experiments) != len(bouts):
            raise ValueError("There has to be one bout dataframe for each experiment dataframe")

        self.is_frame_train = []
        self.stimulus = []
        self.bout_starts = []
        self.bout_ends = []  # this will be our behavioral predictor
        self.static_data = []

        for experiment_data, bout_data in zip(experiments, bouts):
            exp_phase = np.array(experiment_data["Experiment phase"])
            stimulus = np.array(experiment_data["Temperature"])
            bout_starts = np.zeros(stimulus.size)
            bout_ends = np.zeros(stimulus.size)
            bstart_frames = np.array(bout_data["Start"]).astype(int)
            b_end_frames = np.array((bout_data["Stop"])).astype(int)
            bout_starts[bstart_frames] = 1
            bout_ends[b_end_frames] = 1
            # create n_frames x 2 matrix of previous bout indicators
            prev_bout = np.zeros((stimulus.size, 2))
            b_disps = np.array(bout_data["Displacement"])
            b_mags = np.abs(np.array(bout_data["Angle change"]))
            for i, stop in enumerate(b_end_frames):
                static_data = np.c_[b_disps[i], b_mags[i]]
                if i == b_end_frames.size-1:
                    # this is the last one so fill this information until the end
                    prev_bout[stop:, :] = static_data
                else:
                    # fill until the next bout end (inclusive)
                    prev_bout[stop:b_end_frames[i+1]+1, :] = static_data
            # limit to selected phases (note this is somewhat lazy - doing that beforehand would be more efficient)
            for p in phases:
                is_frame_train = np.zeros(np.sum(exp_phase == p), dtype=bool)
                # add test periods into each quarter of the experimental period
                quarter_length = is_frame_train.size // 4
                train_per_quarter = int(quarter_length * train_fraction)
                for q in range(4):
                    is_frame_train[quarter_length*q:quarter_length*q+train_per_quarter] = True
                self.is_frame_train.append(is_frame_train)
                self.stimulus.append(stimulus[exp_phase == p])
                self.bout_starts.append(bout_starts[exp_phase == p])
                self.bout_ends.append(bout_ends[exp_phase == p])
                self.static_data.append(prev_bout[exp_phase == p, :])
        # Note: While input_steps long slices will be constructed for the dynamic model inputs (stimulus and starts)
        # for the static data the values at the output frame to be predicted will be used
        self.input_steps = input_steps
        self.is_frame_train = np.hstack(self.is_frame_train)
        self.stimulus = np.hstack(self.stimulus)
        self.bout_starts = np.hstack(self.bout_starts)
        self.bout_ends = np.hstack(self.bout_ends)
        self.static_data = np.vstack(self.static_data)
        # bin 10-fold to 25Hz final data-rate - we average for temperature and use the maximum for bout_starts (i.e. if
        # there is at least one bout the binned value will be 1 but if there were 2 bouts we will ignore that)
        # for assigning train frames we will also use the maximum
        # to bin the static data we assign the start value in each bin to avoid having the values reflect anything
        # of a potential bout within the bin
        self.is_frame_train = max_bin_1d(self.is_frame_train, 10)
        self.stimulus = mean_bin_1d(self.stimulus, 10)
        self.bout_starts = max_bin_1d(self.bout_starts, 10)
        self.bout_ends = max_bin_1d(self.bout_ends, 10)
        self.static_data = self.static_data[::10, :][:self.bout_starts.size, :]
        if standards is None:
            self.stimulus = safe_standardize(self.stimulus)
            self.bout_ends = safe_standardize(self.bout_ends)
            self.static_data = safe_standardize(self.static_data, axis=0)
            print("Warning: Data is standardized using intrinsic measures")
        else:
            self.stimulus -= standards.stimulus_mean
            self.stimulus /= standards.stimulus_std
            self.bout_ends -= standards.bout_end_mean
            self.bout_ends /= standards.bout_end_std
            self.static_data -= standards.static_mean[None, :]
            self.static_data /= standards.static_std[None, :]
        if shuffle:
            self.bout_starts = np.roll(self.bout_starts, self.bout_starts.size//3)

    def _get_data(self, selector: np.ndarray, batch_size: int):
        out_frames = np.arange(self.input_steps, self.bout_starts[selector].size)
        out_data = self.bout_starts[selector][out_frames].astype(np.float32).copy()
        in_data = np.full((out_data.size, self.input_steps, 2), np.nan).astype(np.float32)
        in_data_static = self.static_data[selector][out_frames-1].copy()

        # create indexing matrix for triggered retrieval - we trigger on the bout occurance itself
        # since our behavioral history consists of bout-ends which should never overlap with starts
        ix_mat = indexing_matrix(out_frames, self.input_steps-1, 0, self.bout_starts[selector].size)[0]
        assert ix_mat.shape[0] == out_data.size

        in_data[:, :, 0] = self.stimulus[selector][ix_mat]
        in_data[:, :, 1] = self.bout_ends[selector][ix_mat]

        ds = tf.data.Dataset.from_tensor_slices((in_data.astype(np.float32), in_data_static.astype(np.float32),
                                                 out_data.astype(np.float32))).\
            shuffle(in_data.shape[0], reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
        return ds.prefetch(tf.data.AUTOTUNE)

    def training_data(self, batch_size=1024):
        """
        Creates training data
        :param batch_size: The training batch size to use
        :return: Tensorflow dataset that can be used for training with randomization
        """
        # NOTE: Since for this network, the output is equivalent to one of the inputs (bout starts), inputs should
        # only contain data up to the last frame before the current output frame and exclude the output frame itself
        if np.sum(self.is_frame_train) == 0:
            # no train set
            return None
        return self._get_data(self.is_frame_train, batch_size)

    def test_data(self, batch_size=1024):
        """
        Creates test data
        :param batch_size: The training batch size to use
        :return: Tensorflow dataset that can be used for testing
        """
        if np.sum(self.is_frame_train) == self.is_frame_train.size:
            # no test set
            return None
        return self._get_data(np.logical_not(self.is_frame_train), batch_size)


def create_overwrite(storage: Union[h5py.File, h5py.Group], name: str, data: Any, overwrite: bool,
                     compress=False) -> None:
    """
    Allows to create a new dataset in an hdf5 file an if desired overvwrite any old data
    :param storage: The hdf5 file or group used to store the information
    :param name: The name of the dataset
    :param data: The data
    :param overwrite: If true any old data with the same name will be deleted and subsequently replaced
    :param compress: If true, data will be stored compressed
    """
    if overwrite and name in storage:
        del storage[name]
    if compress:
        storage.create_dataset(name, data=data, compression="gzip", compression_opts=5)
    else:
        storage.create_dataset(name, data=data)


def bin_1d(arr_in: np.ndarray, bin_frames: int, bin_fun) -> np.ndarray:
    """
    Bins an array setting the value of the output array to value of applying bin_fun to all elements in the bin.
    arr_in will be truncated to a multiple of bin_frames before binning
    :param arr_in: The input array n_frames in size
    :param bin_frames: How many frames should be binned together
    :param bin_fun: Function to apply to each bin. Must accept "axis" argument as it will be applied to 2D intermediate
    :return: n_frames//bin_frames sized binned output array
    """
    s_to_consider = (arr_in.size // bin_frames) * bin_frames
    out_size = arr_in.size // bin_frames
    out = arr_in[:s_to_consider].copy().reshape((out_size, bin_frames))
    return bin_fun(out, axis=1)


def max_bin_1d(arr_in: np.ndarray, bin_frames: int) -> np.ndarray:
    """
    Bins an array setting the value of the output array to the maximum value in each bin. arr_in will be truncated to
    a multiple of bin_frames before binning
    :param arr_in: The input array n_frames in size
    :param bin_frames: How many frames should be binned together
    :return: n_frames//bin_frames sized binned output array
    """
    return bin_1d(arr_in, bin_frames, np.max)


def mean_bin_1d(arr_in: np.ndarray, bin_frames: int) -> np.ndarray:
    """
    Bins an array setting the value of the output array to the average value in each bin. arr_in will be truncated to
    a multiple of bin_frames before binning
    :param arr_in: The input array n_frames in size
    :param bin_frames: How many frames should be binned together
    :return: n_frames//bin_frames sized binned output array
    """
    return bin_1d(arr_in, bin_frames, np.mean)


def safe_standardize(x: np.ndarray, axis: Optional[int] = None, epsilon=1e-9) -> np.ndarray:
    """
    Standardizes an array to 0 mean and unit standard deviation avoiding division by 0
    :param x: The array to standardize
    :param axis: The axis along which standardization should be performmed
    :param epsilon: Small constant to add to standard deviation to avoid divide by 0 if sd(x)=0
    :return: The standardized array of same dimension as x
    """
    if x.ndim == 1 or axis is None:
        y = x - np.mean(x)
        y /= (np.std(y) + epsilon)
    else:
        y = x - np.mean(x, axis=axis, keepdims=True)
        y /= (np.std(y, axis=axis, keepdims=True) + epsilon)
    return y


def indexing_matrix(trigger_frames: np.ndarray, f_past: int, f_future: int,
                    input_length: int) -> Tuple[np.ndarray, int, int]:
    """
    Builds an indexing matrix with length(trigger_frames) rows and columns
    that index out frames from trigger_frames(n)-f_past to
    trigger_frames(n)+f_future. Indices will be clipped at [1...input_length]
    with rows containing frames outside that range removed
    :param trigger_frames: Vector with all the intended trigger frames
    :param f_past: The number of frames before each trigger frame that should be included
    :param f_future: The same for the number of frames after each trigger frame
    :param input_length: The length of the input on which to trigger. Determines
            which rows will be removed from the matrix because they would index out
            non-existent frames
    :return:
        [0]:T he trigger matrix for indexing out all frames
        [1]: The number of rows that have been cut out because they would have contained frames with index < 0
        [2]: The number of rows that have been removed from the back because they would have contained frames
         with index >= input_length
    """
    if trigger_frames.ndim > 1:
        raise ValueError('trigger_frames has to be a vector')

    to_take = np.r_[-1 * f_past:f_future + 1]

    # turn trigger_frames into a size 2 array consisting of 1 column only
    trig_f = np.expand_dims(trigger_frames, 1)
    # turn to_take into a size 2 array consisting of one row only
    to_take = np.expand_dims(to_take, 0)
    # now we can use repeat to construct matrices:
    index_mat = np.repeat(trig_f, to_take.size, 1) + np.repeat(to_take, trig_f.size, 0)
    # identify front and back rows that need to be removed
    cut_front = np.sum(np.sum(index_mat < 0, 1, dtype=float) > 0, 0)
    cut_back = np.sum(np.sum(index_mat >= input_length, 1, dtype=float) > 0, 0)
    # remove out-of-bounds rows and return - if statement (seems) necessary since
    # there is no -0 for indexing the final frame
    if cut_back > 0:
        return index_mat[cut_front:-1 * cut_back, :].astype(int), cut_front, cut_back
    else:
        return index_mat[cut_front::, :].astype(int), cut_front, cut_back


@tf.function
def dca_dr(mdl: model_defs.BoutProbability, dyn_input: np.ndarray,
           stat_input: np.ndarray, training: bool) -> tf.Tensor:
    x_dynamic = tf.convert_to_tensor(dyn_input)
    x_static = tf.convert_to_tensor(stat_input)
    with tf.GradientTape() as t1:
        t1.watch(x_dynamic)
        bout_logit = mdl(x_dynamic, x_static, training=training)
    jacobian_dynamic = t1.gradient(bout_logit, x_dynamic)
    return jacobian_dynamic


def roc_auc_prob_model(data: tf.data.Dataset, model: model_defs.BoutProbability) -> float:
    """
    Computes the fraction of times a given model ranks the swim probability on a random true swim frame higher
    than the probability on a random non-swim frame by calculating the ROC-AUC of the classifier.
    :param data: Tensorflow dataset on which to evaluate
    :param model: The model to test
    :return: roc_auc
    """
    bout_out = []
    bout_prob_pred = []
    for row in data:
        if len(row) == 3:
            # for a bout probability model the data is bout-probability data
            inp_dyn, inp_stat, outp = row
        else:
            # Bout flip removed until the model's data-structure is redefined
            assert False
            # # for turn-flip the data is bout-feature data
            # inp_dyn, inp_stat = row[:2]
            # outp = row[-1]
        bout_out.append(outp.numpy())
        pred = model.get_probability(inp_dyn, inp_stat)
        bout_prob_pred.append(pred)
    bout_out = np.hstack(bout_out)
    bout_prob_pred = np.hstack(bout_prob_pred)
    return roc_auc_score(bout_out > 0.5, bout_prob_pred)


def boot_data(data_in: np.ndarray, n_boot: int, boot_fun: Any) -> np.ndarray:
    """
    Performs bootstrapping of the input data, sampling with replacement along axis 0
    :param data_in: n_samples x n_features data array for bootstrapping
    :param n_boot: The number of bootstrap samples to generate
    :param boot_fun: The function to apply to each variate (e.g. numpy.mean), must take axis parameter
    :return: n_boot x n_features array of bootstrap samples
    """
    indices = np.arange(data_in.shape[0])
    boot_out = np.zeros((n_boot, data_in.shape[1]))
    for i in range(n_boot):
        chosen = np.random.choice(indices, indices.size, replace=True)
        boot_out[i, :] = boot_fun(data_in[chosen, :], axis=0)
    return boot_out


def boot_error(data_in: np.ndarray, n_boot: int, boot_fun: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs bootstrapping of the input data, sampling with replacement along axis 0 and only returns mean and
    standard error
    :param data_in: n_samples x n_features data array for bootstrapping
    :param n_boot: The number of bootstrap samples to generate
    :param boot_fun: The function to apply to each variate (e.g. numpy.mean), must take axis parameter
    :return:
        [0]: n_features array of bootstrap mean
        [1]: n_features array of bootstrap standard error
    """
    boot = boot_data(data_in, n_boot, boot_fun)
    return np.mean(boot, axis=0), np.std(boot, axis=0)


def modelweights_to_hdf5(storage: Union[h5py.File, h5py.Group], m_weights: List[np.ndarray]) -> None:
    """
    Stores tensorflow weights of a model sequentially to an hdf5 file or group to allow for more compact storage
    across many models (NOTE: No states are saved)
    :param storage: The hdf5 file or group used to store the information
    :param m_weights: List of weight arrays returned by keras.model.get_weights()
    """
    storage.create_dataset("n_layers", data=len(m_weights))
    for i, mw in enumerate(m_weights):
        if type(mw) == np.ndarray:
            storage.create_dataset(f"layer_{i}", data=mw, compression="gzip", compression_opts=5)
        else:
            storage.create_dataset(f"layer_{i}", data=mw)


def modelweights_from_hdf5(storage: Union[h5py.File, h5py.Group]) -> List[np.ndarray]:
    """
    Loads tensorflow model weights from an hdf5 file or group
    :param storage: The hdf5 file or group from which to load the information
    :return: List of weight arrays that can be used with keras.model.set_weights() to set model weights
    """
    n_layers = storage["n_layers"][()]
    m_weights = []
    for i in range(n_layers):
        m_weights.append(storage[f"layer_{i}"][()])
    return m_weights


def count_turn_streaks(turn_dirs: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Counts the occurence of different lengths of "maintain" and "alternating" turn direction streaks
    :param turn_dirs: Vector of successive turn directions (1/-1)
    :param max_len: The maximum streak length to consider
    :return:
        [0]: max_len sized vector of counts of maintain streaks - note the last entry contains max_len or longer!
        [1]: max_len sized vector of counts of alternating streaks - note the last entry contains max_len or longer!
    """
    m_vec = np.zeros(max_len, dtype=int)
    a_vec = np.zeros(max_len, dtype=int)
    for i in range(turn_dirs.size - max_len):
        # start finding streaks starting at i
        init_dir = turn_dirs[i]  # important to track maintain streaks
        last_dir = turn_dirs[i]  # important to track alternate streaks
        in_maintain = True  # indicates if currently a maintain streak is present
        in_alternate = True  # indicates if currently a alternating streak is present
        maintain_count = 0
        alternate_count = 0
        for k in range(i+1, i+max_len):
            if turn_dirs[k] != init_dir:
                in_maintain = False
            elif in_maintain:
                maintain_count += 1
            if turn_dirs[k] == last_dir:
                in_alternate = False
            elif in_alternate:
                alternate_count += 1
            last_dir = turn_dirs[k]
            if not in_maintain and not in_alternate:
                break
        m_vec[maintain_count] += 1
        a_vec[alternate_count] += 1
    return m_vec, a_vec


def mean_fill_nan(a: np.ndarray) -> np.ndarray:
    """
    Fills nan-values in an array. If at the beginning or end, will be filled with the nanmean of the entire array.
    In-between NaN values will be filled with the average of the first and last non-nan values bordering the gap
    :param a: The array to fill
    :return: New array with NaN values filled
    """
    isnan = np.isnan(a)
    starts_ends = np.diff(np.r_[0, isnan])
    starts = np.where(starts_ends == 1)[0]
    ends = np.where(starts_ends == -1)[0]
    assert ends.size <= starts.size <= ends.size + 1
    if ends.size < starts.size:
        a[starts[-1]:] = np.nanmean(a)
        starts = starts[:-1]
    assert ends.size == starts.size
    ret = a.copy()
    for s, e in zip(starts, ends):
        if s == 0:
            ret[s:e] = np.nanmean(a)
        else:
            ret[s:e] = (a[s-1] + a[e]) / 2
    return ret


def get_flip_maintain(angles: np.ndarray, window_length: int) -> np.ndarray:
    """
    Takes an angle trace and turns it into a trace of 0(flip) and 1(maintain) based on whether the current turn
    is in the same direction as the average direction of the previous window_length turns
    :param angles: The bout turn angles
    :param window_length: The length of the window into the past for average direction matching to call "maintain"
    :return: angles.size array of flip(0) and maintain(1) indicators
    """
    if window_length < 2:
        raise ValueError("window_length must be greater than 1")
    flip_maintain = np.full_like(angles, np.nan)
    # for each swim at time t compute the average turn angle from itself and window_length-1 into the past
    # that way, for a bout at time t we can look at the value of avg_dir at t-1 to determine if it was
    # a flip or a maintain
    avg_dir = np.convolve(angles, np.ones(window_length)/window_length)[:angles.size]
    for i in range(1, angles.size):
        if np.sign(angles[i]) == np.sign(avg_dir[i-1]):
            flip_maintain[i] = 1
        else:
            flip_maintain[i] = 0
    return flip_maintain


def ks_bootstrap_test_by_fish(sample1: List, sample2: List, nboot: int) -> Tuple[float, np.ndarray, float]:
    """
    Computes ks test statistic in cases where data come from different individuals and where variability is believed
    to be larger between individuals than within individuals
    :param sample1: List of data-samples with each element in the list being one individual fish
    :param sample2: List of data-samples with each element in the list being one individual fish
    :param nboot: The number of bootstrap resamples to perform
    :return:
        [0]: The p-value
        [1]: The test statistic of the joint bootstrap resamples
        [2]: The test statistic of the true comparison
    """
    true_ks_stat = kstest(np.hstack(sample1), np.hstack(sample2)).statistic
    combined = sample1 + sample2
    ix_combined = np.arange(len(combined)).astype(int)
    boot_stats = np.zeros(nboot)
    for i in range(nboot):
        ix1 = np.random.choice(ix_combined, len(sample1))
        ix2 = np.random.choice(ix_combined, len(sample2))
        s1 = np.hstack([combined[ix] for ix in ix1])
        s2 = np.hstack([combined[ix] for ix in ix2])
        boot_stats[i] = kstest(s1, s2).statistic
    # estimate p-value based on normal approximation if no elements in boot_stats are larger than true_ks_stat
    if np.sum(boot_stats >= true_ks_stat) == 0:
        print("p-value estimated through normal approximation")
        std = np.std(boot_stats)
        n_sigma = (true_ks_stat - np.mean(boot_stats)) / std
        p = 1 - norm.cdf(n_sigma)
    else:
        p = np.sum(boot_stats >= true_ks_stat) / nboot
    return p, boot_stats, true_ks_stat


if __name__ == '__main__':
    pass
