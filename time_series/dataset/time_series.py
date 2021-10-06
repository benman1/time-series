"""Time-series data classes."""
from dataclasses import dataclass
import datetime
import logging
from typing import Optional, Sequence

import numpy as np
from numpy import typing as np_types
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.tsatools import lagmat

from time_series.dataset import Dataset


LOGGER = logging.getLogger(__file__)


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df=pd.DataFrame,
        val_df=Optional[pd.DataFrame],
        test_df=Optional[pd.DataFrame],
        label_columns: Optional[Sequence[str]] = None,
    ):
        """Create a window generator for the time-series.

        Adapted from: https://www.tensorflow.org/tutorials/structured_data/time_series

        Parameters
        ----------
        input_width: how many columns
        label_width: how many columns to predict (should be same as input_width)
        shift: how many lags to work with
        train_df: the dataframe, where the time-series come from. Each row should be a
            time step, and each column is a variable that varies over time.
        val_df: dataframe for validation.
        test_df: dataframe for testing.
        label_columns: this can be used for plotting.

        The data will come as shape [(None, input_width, shift)]
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffle: bool = True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,
        )
        return ds.map(self.split_window)

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


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


class MockTs(TimeSeries):
    """This class generates 'mock' time series data."""

    def __init__(self, dimensions: int = 1, batch_size: int = 1, n_steps: int = 10):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.n_steps = n_steps
        data = pd.DataFrame(
            {f"col_{i}": self.generate_time_series() for i in range(self.dimensions)}
        )
        self.ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.n_steps,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )

    @staticmethod
    def generate_time_series(freq: float = 365.2425 * 24 * 60 * 60):
        date = pd.date_range(
            start=datetime.date(
                2016, np.random.randint(1, 12), np.random.randint(1, 29)
            ),
            periods=800,
            freq="D",
        )
        return np.random.randint(1, 10000) * (
            1 + np.sin(date.astype("int64") // 1e9 * (2 * np.pi / freq))
        )

    def __next__(self):
        for batch in self.ds:
            yield batch

    def __iter__(self):
        return self


def sample_to_input(
    sample: pd.DataFrame, lag: int, two_dim: bool = False
) -> np_types.ArrayLike:
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
    """This is in place for a generator.

    Create lags and split between train and test.

    Attributes:
         lag, train_split, X_train, y_train, X_test, y_test.
    """

    X_train: np_types.ArrayLike
    y_train: np_types.ArrayLike
    X_test: np_types.ArrayLike
    y_test: np_types.ArrayLike

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
        return self.X_train.shape[2]

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
    def exo_dim(self):
        """This class doesn't handle exogenous attributes."""
        return 0

    @property
    def horizon(self):
        """How many steps to forecast to?"""
        return self.y_train.shape[1] if not self.two_dim else 1
