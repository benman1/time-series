from functools import partial
import logging
import numpy as np

from tensorflow.keras.layers import Input, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from deepar.model.loss import gaussian_likelihood
from deepar.model import NNModel
from deepar.model.layers import GaussianLayer


logger = logging.getLogger("deepar")


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
                dimensions=self.ts_obj.dimensions
            )
        self._output_layer_name = "main_output"
        self.get_intermediate = None

    @staticmethod
    def basic_structure(n_steps=20, dimensions=1):
        """
        This is the method that needs to be patched when changing NN structure
        :return: inputs_shape (tuple), inputs (Tensor), [loc, scale] (a list of theta parameters
        of the target likelihood)
        """
        input_shape = (n_steps, dimensions)
        inputs = Input(shape=input_shape)
        x = LSTM(4, return_sequences=True)(inputs)
        x = Dense(3, activation="relu")(x)
        loc, scale = GaussianLayer(dimensions, name="main_output")(x)
        return input_shape, inputs, [loc, scale]

    def fit(self, verbose=False):
        self.keras_model.fit(
            ts_generator(self.ts_obj, self.ts_obj.n_steps),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
        )
        if verbose:
            logger.debug("Model was successfully trained")
        self.get_intermediate = K.function(
            inputs=[self.keras_model.input],
            outputs=self.keras_model.get_layer(self._output_layer_name).output,
        )

    def instantiate_and_fit(self, verbose=False):
        input_shape, inputs, theta = self.nn_structure()
        model = Model(inputs, theta[0])
        model.compile(loss=self.loss(theta[1]), optimizer=self.optimizer)
        self.keras_model = model
        self.fit(verbose)

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
