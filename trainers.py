import numpy as np
import utility
import model_defs as md
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import pandas as pd


class Trainer(ABC):
    """
    Abstract base class for model training/testing classes
    """
    def __init__(self, history: int, std: utility.InputDataStandard, all_fish: List[pd.DataFrame],
                 all_bouts: List[pd.DataFrame], train_fraction: float, data_class: callable, shuffle: bool):
        self.history = history
        self.standard = std
        # use the provided class to create our training and test data
        self.data = data_class(self.history, all_fish, all_bouts, (9,), train_fraction, self.standard, shuffle=shuffle)
        self.model: Optional[md.SwimBaseModel] = None

    @abstractmethod
    def train(self, n_epochs: int) -> None:
        """
        Trains the underlying model using the underlying data
        :param n_epochs: The number of training epochs to run
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        """
        Uses the test data and the underlying model to generate model test predictions
        """
        raise NotImplementedError()

    def get_model_weights(self) -> Optional[List[np.ndarray]]:
        """
        Get the current model weights
        """
        if self.model is not None:
            return self.model.get_weights()
        else:
            return None


class BoutProbabilityTrainer(Trainer):
    """
    Trainer class for bout probability models
    """
    def __init__(self, history: int, batch_size: int, std: utility.InputDataStandard, all_fish: List[pd.DataFrame],
                 all_bouts: List[pd.DataFrame], train_fraction: float, shuffle: bool, l1: Optional[float] = None):
        """
        Create a new BoutProbabilityTrainer instance
        :param history: The length of the model history
        :param batch_size: The training batch size
        :param std: The data standardization object
        :param all_fish: All fish data
        :param all_bouts: All bout data
        :param train_fraction: The fraction of frames in each experiment in the training data
        :param shuffle: If set to true, rotate outputs with respect to inputs by 1/3 of the data-length
        :param l1: The L1 regularization parameter
        """
        super().__init__(history, std, all_fish, all_bouts, train_fraction, utility.Data_BoutProbability, shuffle)
        self.batch_size = batch_size
        self.model = md.get_standard_boutprob_model(self.history, l1)
        # initialize model-weights
        test_inps = np.random.randn(1, history, 2)
        test_stat_inps = np.random.randn(1, 2)
        self.model(test_inps, test_stat_inps)

    def train(self, n_epochs: int) -> None:
        """
        Trains the underlying model using the underlying data
        :param n_epochs: The number of training epochs to run
        """
        md.train_boutprob_model(self.model, self.data.training_data(self.batch_size), n_epochs)

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtain real outputs and test predictions of the model
        :return:
            [0]: The true bout outputs
            [1]: The model probability predictions
        """
        bout_out = []
        bout_prob_pred = []
        data = self.data.test_data(self.batch_size)
        for inp_dyn, inp_stat, outp in data:
            bout_out.append(outp.numpy())
            pred = self.model.get_probability(inp_dyn, inp_stat)
            bout_prob_pred.append(pred)
        bout_out = np.hstack(bout_out)
        bout_prob_pred = np.hstack(bout_prob_pred)
        return bout_out, bout_prob_pred


if __name__ == '__main__':
    pass
