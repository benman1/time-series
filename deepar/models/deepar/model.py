"""DeepAR model.

Based on https://github.com/arrigonialberto86/deepar
By Alberto Arrigoni.
"""
from functools import partial
import logging
from typing import Optional, Union

import numpy as np
from numpy.random import normal
import pandas as pd

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model

from deepar.dataset.time_series import TrainingDataSet
from deepar.models.deepar.loss import gaussian_likelihood
from deepar.models import NNModel
from deepar.models.deepar.layers import GaussianLayer


LOGGER = logging.getLogger(__name__)


class DeepAR(NNModel):
    """DeepAR model."""

    def __init__(
        self,
        data: TrainingDataSet,
        epochs=100,
        loss=gaussian_likelihood,
        optimizer: str = "adam",
    ):
        """Init.

        Arguments:
            df (pd.DataFrame): a dataframe of shape time x value columns
            epochs: how many epochs to train (initially).
            loss: a loss function.
            optimizer: which optimizer to use.
        """
        self.data = data
        self.inputs, self.z_sample = None, None
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model: Optional[Model] = None
        self.nn_structure = partial(
            DeepAR.basic_structure, n_steps=data.n_steps, dimensions=data.dimensions
        )
        self._output_layer_name = "main_output"
        self.gaussian_layer: Optional[Model] = None

    @staticmethod
    def basic_structure(n_steps=20, dimensions=1):
        """
        This is the method that needs to be patched when changing NN structure
        :return: inputs_shape (tuple), inputs (Tensor), [loc, scale] (a list of theta parameters
        of the target likelihood).

        Please note that I've made up scaling rules of the hidden layer dimensions.
        """
        input_shape = (n_steps, dimensions)
        inputs = Input(shape=input_shape)
        x = LSTM(
            4,  # int(4 * (1 + math.pow(math.log(dimensions), 4))),
            return_sequences=True,
            dropout=0.1,
        )(inputs)
        # int(4 * (1 + math.log(dimensions))),
        x = Dense(4, activation="relu")(x)
        loc, scale = GaussianLayer(dimensions, name="main_output")(x)
        return input_shape, inputs, [loc, scale]

    def fit(
        self,
        epochs: Optional[int] = None,
        verbose: Union[str, int] = "auto",
        patience: int = 10,
    ):
        """Fit models.

        This is called from instantiate and fit().

        Args:
            epochs (Optional[int]): number of epochs to train. If nothing
                defined, take self.epochs. Please the early stopping (patience).
            verbose (Union[str, int]): passed to keras.fit(). Can be
                "auto", 0, or 1.
            patience (int): Number of epochs without without improvement to stop.
        """
        from tensorflow.python.framework.ops import disable_eager_execution

        disable_eager_execution()

        from tensorflow.compat.v1.experimental import output_all_intermediates

        output_all_intermediates(True)

        if not epochs:
            epochs = self.epochs
        self.keras_model.fit(
            self.data.X_train,
            self.data.y_train,
            epochs=epochs,
            verbose=verbose,
            callbacks=self.callbacks,
        )
        if verbose:
            LOGGER.debug("Model was successfully trained")

    def build_model(self):
        input_shape, inputs, theta = self.nn_structure()
        self.keras_model = Model(inputs, theta[0])
        LOGGER.info(self.keras_model.summary())
        self.gaussian_layer = Model(
            self.keras_model.input,
            self.keras_model.get_layer(self._output_layer_name).output,
        )
        self.keras_model.compile(loss=self.loss(theta[1]), optimizer=self.optimizer)
        self.gaussian_layer.compile(loss="mse", optimizer="adam")

    def instantiate_and_fit(self, do_fit: bool = True, **fit_kwargs):
        """Compile and train models."""
        self.build_model()
        if do_fit:
            self.fit(**fit_kwargs)

    @property
    def model(self):
        return self.keras_model

    def predict_theta_from_input(self, input_list):
        """Predict from GaussianLayer.

        This function takes an input of size equal to the n_steps specified in 'Input' when building the
        network.
        :param input_list:
        :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
        corresponding to [[mu_values], [sigma_values]]
        """
        if not self.keras_model.history:
            raise ValueError("Model must be trained first!")

        return self.gaussian_layer.predict(input_list)

    def get_sample_prediction(self, sample_df: pd.DataFrame):
        """WIP."""
        self.ts_obj.test_df = sample_df
        sample = self.ts_obj.test
        output = self.predict_theta_from_input(sample)
        samples = []
        for mu, sigma in zip(output[0].reshape(-1), output[1].reshape(-1)):
            sample = normal(
                loc=mu, scale=np.sqrt(sigma), size=1
            )  # self.ts_obj.dimensions)
            samples.append(sample)

        return np.array(samples).reshape(
            (self.ts_obj.label_width, self.ts_obj.dimensions)
        )


if __name__ == "__main__":
    """For debugging."""
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    from tensorflow.compat.v1.experimental import output_all_intermediates

    output_all_intermediates(True)

    from deepar.dataset.utils import get_energy_demand

    train_df = get_energy_demand()

    dp_model = DeepAR(train_df, epochs=10)
    dp_model.instantiate_and_fit(verbose=1, epochs=1, patience=1)
