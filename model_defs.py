"""
Module for all network models with tensorflow dependency
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, initializers
from typing import Optional, Union, Tuple, List


class NotInitialized(Exception):
    def __init__(self, message):
        super().__init__(message)


class SwimBaseModel(keras.Model):
    """
    Base class for bout models
    """

    def __init__(self, n_units: int, n_conv: int, drop_rate: float, input_length: int, activation: str, n_deep: int):
        """
        Creates a new model
        :param n_units: The number of units in each dense layer
        :param n_conv: The number of units in each initial convolutional layer
        :param drop_rate: The drop-out rate during training
        :param input_length: The length (across time) of inputs to the network (sets conv filter size)
        :param activation: The activation function to use
        :param n_deep: The number of dense layers
        """
        if n_deep < 1:
            raise ValueError("Need at least one deep layer")
        super(SwimBaseModel, self).__init__()
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("drop_rate has to be between 0 and 1")
        if n_units < 1:
            raise ValueError("Need at least one unit in each dense layer")
        if n_conv < 1:
            raise ValueError("Need at least one convolutional unit")
        self._n_units: int = n_units
        self._n_conv: int = n_conv
        self._n_deep: int = n_deep
        self.input_length: int = input_length
        self._activation: str = activation
        self._drop_rate: float = drop_rate
        self.l2_constraint: float = 2e-4  # sparsity constraint on weight vectors
        self.learning_rate: float = 1e-3
        # optimizer and loss functions
        self.optimizer: Optional[keras.optimizers.Optimizer] = None
        self._initialized: bool = False
        # layers
        self._conv_layer: Optional[keras.layers.Layer] = None  # Convolutional layer
        self._drop_cl: Optional[keras.layers.Layer] = None  # Dropout of convolutional layer
        self._all_deep: Optional[List[keras.layers.Layer]] = None  # List of model's deep layers
        self._flatten: Optional[keras.layers.Layer] = None

    def setup(self) -> None:
        """
        Initializes the model, resetting weights
        """
        # processing
        self._conv_layer = layers.Conv1D(filters=self.n_conv,
                                         kernel_size=self.input_length,
                                         # set kernel size = input size => computes dot product
                                         use_bias=False,  # for simplicity omit bias from convolutional layers
                                         padding='valid',
                                         activation=None,
                                         kernel_initializer=initializers.GlorotUniform(),
                                         kernel_regularizer=regularizers.l2(self.l2_constraint),
                                         strides=1, name="Convolution")
        self._flatten = layers.Flatten()
        self._drop_cl = layers.Dropout(self.drop_rate)
        self._all_deep = []
        for i in range(self._n_deep):
            dense = layers.Dense(units=self.n_units, activation=self.activation,
                                 kernel_initializer=initializers.GlorotUniform(),
                                 kernel_regularizer=regularizers.l2(self.l2_constraint), name=f"Deep{i}")
            self._all_deep.append(dense)
            drop = layers.Dropout(self.drop_rate)
            self._all_deep.append(drop)
        # create our optimizer and loss functions
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

    def clear_model(self) -> None:
        """
        Clears and uninitializes the model
        """
        self._conv_layer = None  # Convolutional layer
        self._drop_cl = None  # Dropout of convolutional layer
        self._all_deep = None  # First deep layer
        self._initialized = False

    def check_init(self) -> None:
        if not self._initialized:
            raise NotInitialized("Model not initialized. Call setup or load.")

    def check_input(self, inputs, static_inputs) -> None:
        if inputs.shape[1] != self.input_length:
            raise ValueError("Input length across time different than expected")

    @tf.function
    def call(self, inputs: Union[np.ndarray, tf.Tensor], static_inputs: Union[np.ndarray, tf.Tensor],
             training: Optional[bool] = None, mask=None) -> tf.Tensor:
        if training is None:
            training = False
        self.check_init()
        inputs = self._conv_layer(inputs, training=training)
        inputs = self._drop_cl(inputs, training=training)
        inputs = self._flatten(inputs)
        for layer in self._all_deep:
            inputs = layer(inputs, training=training)
        return inputs

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def drop_rate(self) -> float:
        return self._drop_rate

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def n_conv(self) -> int:
        return self._n_conv

    @property
    def conv_layer(self) -> Optional[keras.layers.Layer]:
        return self._conv_layer

    @property
    def n_deep(self) -> int:
        return self.n_deep

    @property
    def flatten(self) -> Optional[keras.layers.Layer]:
        return self._flatten


class BoutProbability(SwimBaseModel):
    """
    Network to predict the probability of swim-bouts based on dynamic and static input features
    """

    def __init__(self, n_units: int, n_conv: int, drop_rate: float, input_length: int, activation: str):
        """
        Creates a new BoutProbability model
        :param n_units: The number of units in each dense layer
        :param n_conv: The number of units in each initial convolutional layer
        :param drop_rate: The drop-out rate during training
        :param input_length: The length (across time) of inputs to the network (sets conv filter size)
        :param activation: The activation function to use
        """
        super(BoutProbability, self).__init__(n_units, n_conv, drop_rate, input_length, activation, 2)
        # loss function
        self.loss_fn: Optional[keras.losses.Loss] = None
        # output placeholder
        self._out: Optional[keras.layers.Layer] = None

    def setup(self) -> None:
        """
        Initializes the model, resetting weights
        """
        super(BoutProbability, self).setup()
        # output: This is just one value, that represents the log-odds of making a bout (logit)
        self._out = layers.Dense(1, activation=None, name="Out")
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self._initialized = True

    def get_output(self, inputs: np.ndarray, static_inputs: np.ndarray) -> float:
        """
        Returns the log-probability given the model inputs
        :param inputs: batchsize x input_length x n_regressors (the channels)
        :param static_inputs: batchsize x 3 input of previous bout features (displacement, t_magnitude, t_direction)
        :return: 1 output value corresponding to log probability of a bout occuring
        """
        self.check_input(inputs, static_inputs)
        out = self(inputs, static_inputs)
        return out.numpy().ravel()

    def get_probability(self, inputs: np.ndarray, static_inputs: np.ndarray, use_drop=False) -> float:
        self.check_input(inputs, static_inputs)
        logit_out = self(inputs, static_inputs, training=use_drop)
        return tf.math.sigmoid(logit_out).numpy().ravel()

    def clear_model(self) -> None:
        """
        Clears and uninitializes the model
        """
        super(BoutProbability, self).clear_model()
        self._out = None

    @tf.function
    def call(self, inputs: Union[np.ndarray, tf.Tensor], static_inputs: Union[np.ndarray, tf.Tensor],
             training: Optional[bool] = None, mask=None) -> tf.Tensor:
        inputs = super(BoutProbability, self).call(inputs, static_inputs, training, mask)
        return self._out(inputs)

    @tf.function
    def train_step(self, btch_inputs: Union[np.ndarray, tf.Tensor], btch_stat_inputs: Union[np.ndarray, tf.Tensor],
                   btch_labels: Union[np.ndarray, tf.Tensor]) -> None:
        """
        Runs one training step on the model
        :param btch_inputs: The time-varying inputs of the batch
        :param btch_stat_inputs: The static features of the last bout
        :param btch_labels: True values signaling bout/no-bout
        """
        with tf.GradientTape() as tape:
            pred = self(btch_inputs, btch_stat_inputs, training=True)
            loss = self.loss_fn(btch_labels, pred)
            loss += sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @property
    def out(self) -> Optional[keras.layers.Layer]:
        return self._out


def train_boutprob_model(mdl: BoutProbability, tset: tf.data.Dataset, n_epochs: int) -> None:
    for e in range(n_epochs):
        for inp, sinp, outp in tset:
            mdl.train_step(inp, sinp, outp)


def get_standard_boutprob_model(hist_steps: int, l2_constraint: Optional[float] = None) -> BoutProbability:
    """
    Creates and returns a BoutProbability instance with standard parameters
    :param hist_steps: The number of history steps in the model
    :param l2_constraint: Optionally set a different l1 sparsity constraint
    :return: The model
    """
    if l2_constraint is None:
        l2_constraint = 1e-5
    m = BoutProbability(64, 20, 0.5, hist_steps, "swish")
    m.learning_rate = 1e-3
    m.l2_constraint = l2_constraint
    m.setup()
    return m


if __name__ == '__main__':
    pass
