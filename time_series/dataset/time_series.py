"""Time-series data classes."""
from packaging import version
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

if version.parse(np.__version__) >= version.parse("1.20.0"):
    from numpy.typing.np_types import ArrayLike
else:
    from typing import Union

    ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

import tensorflow as tf
from statsmodels.tsa.tsatools import lagmat

from time_series.dataset import Dataset


LOGGER = logging.getLogger(__file__)


class TimeSeries(Dataset):
    def __init__(
        self, pandas_df: pd.DataFrame, n_steps: int = 1, batch_size: int = 10,
    ):
        super().__init__()
        assert isinstance(
            pandas_df, (pd.Series, pd.DataFrame)
        ), "Must provide a Pandas df to instantiate this class"
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.dimensions = len(pandas_df.columns)

        data = np.array(pandas_df, dtype=np.float32)
        self.ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.n_steps,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def __next__(self):
        """Iterator."""
        return self.ds.next()


def sample_to_input(sample: pd.DataFrame, lag: int, two_dim: bool = False) -> ArrayLike:
    """Reshape a time-series to be suitable for the models.

    Arguments:
        sample (pd.DataFrame): time x value columns.
        lag (int): the number of previous steps to use as predictors.
        two_dim (bool): whether to reshape as 2D (default 3D)
    Output:
        points x time/lag x columns or (for 2D) time x (columns*lag)
    """
    in_dim = sample.shape[1]
    # drop rows with unknown values both at beginning and end
    if two_dim:
        return lagmat(sample.values, maxlag=lag, trim="both")
    else:
        return np.concatenate(
            [
                np.expand_dims(
                    lagmat(sample.values[:, i], maxlag=lag, trim="both"), axis=2
                )
                for i in range(in_dim)
            ],
            axis=2,
        )


@dataclass
class TrainingDataSet:
    """Utility class that can be used for training and testing.

    Create lags and split between train and test.

    Attributes:
         lag, train_split, X_train, y_train, X_test, y_test.
    """

    X_train: ArrayLike
    y_train: ArrayLike
    X_test: ArrayLike
    y_test: ArrayLike

    def __init__(
        self,
        df: pd.DataFrame,
        lag: int = 10,
        train_split: float = 0.8,
        two_dim: bool = False,
    ):
        self.lag = lag
        self.train_split = train_split
        self.two_dim = two_dim
        lagged = sample_to_input(df, lag, two_dim=two_dim)
        y = np.roll(lagged, shift=-lag, axis=0)
        split_point = int(len(df) * train_split)  # points for training
        self.X_train, self.X_test = (
            lagged[:split_point, ...],
            lagged[split_point:, ...],
        )
        self.y_train, self.y_test = (
            y[:split_point, ...],
            y[split_point:, ...],
        )

    @property
    def n_steps(self):
        """How many steps (lags) to use as predictors."""
        return self.X_train.shape[1] if not self.two_dim else 1

    @property
    def dimensions(self):
        """Number of dimensions."""
        return self.X_train.shape[-1]

    @property
    def n_classes(self):
        """Number of classes.

        This is appropriate for classification tasks.
        """
        return len(np.unique(self.y_train))

    @property
    def input_shape(self):
        """The input shape for a model."""
        return self.X_train.shape[1:]

    @property
    def output_shape(self):
        """The input shape for a model."""
        return self.y_train.shape[1:]

    @property
    def exo_dim(self):
        """This class doesn't handle exogenous attributes."""
        return 0

    @property
    def horizon(self):
        """How many steps to forecast to?"""
        return self.y_train.shape[1] if not self.two_dim else 1
