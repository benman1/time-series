import math
from functools import partial
import logging
from typing import Optional, Union

import numpy as np
from numpy.random import normal

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from deepar.model.loss import gaussian_likelihood
from deepar.model import NNModel
from deepar.model.layers import GaussianLayer


logger = logging.getLogger(__name__)


class DeepAR(NNModel):
    def __init__(
        self,
        ts_obj,
        steps_per_epoch=50,
        epochs=100,
        loss=gaussian_likelihood,
        optimizer="adam",
        with_custom_nn_structure=None,
    ):
        """Init."""

        self.ts_obj = ts_obj
        self.inputs, self.z_sample = None, None
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model = None
        if with_custom_nn_structure:
            self.nn_structure = with_custom_nn_structure
        else:
            self.nn_structure = partial(
                DeepAR.basic_structure,
                n_steps=self.ts_obj.n_steps,
                dimensions=self.ts_obj.dimensions,
            )
        self._output_layer_name = "main_output"
        self.get_intermediate = None

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
            int(4 * (1 + math.pow(math.log(dimensions), 4))),
            return_sequences=True,
            dropout=0.1,
        )(inputs)
        x = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(x)
        loc, scale = GaussianLayer(dimensions, name="main_output")(x)
        return input_shape, inputs, [loc, scale]

    def fit(
        self,
        epochs: Optional[int] = None,
        verbose: Union[str, int] = "auto",
        patience: int = 10,
    ):
        """Fit model.

        This is called from instantiate and fit().

        Args:
            epochs (Optional[int]): number of epochs to train. If nothing
                defined, take self.epochs. Please the early stopping (patience).
            verbose (Union[str, int]): passed to keras.fit(). Can be
                "auto", 0, or 1.
            patience (int): Number of epochs without without improvement to stop.
        """
        if not epochs:
            epochs = self.epochs
        callback = callbacks.EarlyStopping(monitor="loss", patience=patience)
        self.keras_model.fit(
            ts_generator(self.ts_obj, self.ts_obj.n_steps),
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback],
        )
        if verbose:
            logger.debug("Model was successfully trained")
        self.get_intermediate = K.function(
            inputs=[self.keras_model.input],
            outputs=self.keras_model.get_layer(self._output_layer_name).output,
        )

    def build_model(self):
        input_shape, inputs, theta = self.nn_structure()
        model = Model(inputs, theta[0])
        model.compile(loss=self.loss(theta[1]), optimizer=self.optimizer)
        self.keras_model = model

    def instantiate_and_fit(
        self,
        epochs: Optional[int] = None,
        verbose: Union[str, int] = "auto",
        do_fit: bool = True,
    ):
        """Compile and train model."""
        self.build_model()
        self.fit(verbose=verbose, epochs=epochs)

    @property
    def model(self):
        return self.keras_model

    def predict_theta_from_input(self, input_list):
        """
        This function takes an input of size equal to the n_steps specified in 'Input' when building the
        network
        :param input_list:
        :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
        corresponding to [[mu_values], [sigma_values]]
        """
        if not self.get_intermediate:
            raise ValueError("TF model must be trained first!")

        return self.get_intermediate(input_list)

    def get_sample_prediction(self, sample):
        sample = np.array(sample).reshape(
            (1, self.ts_obj.n_steps, self.ts_obj.dimensions)
        )
        output = self.predict_theta_from_input([sample])
        samples = []
        for mu, sigma in zip(output[0].reshape(-1), output[1].reshape(-1)):
            sample = normal(
                loc=mu, scale=np.sqrt(sigma), size=1
            )  # self.ts_obj.dimensions)
            samples.append(sample)
        return np.array(samples).reshape((self.ts_obj.n_steps, self.ts_obj.dimensions))


def ts_generator(ts_obj, n_steps):
    """
    This is a util generator function for Keras
    :param ts_obj: a Dataset child class object that implements the 'next_batch' method
    :param n_steps: parameter that specifies the length of the net's input tensor
    :return:
    """
    while 1:
        batch = ts_obj.next_batch(1, n_steps)
        yield batch[0], batch[1]
