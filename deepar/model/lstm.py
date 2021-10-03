from functools import partial
import logging
from typing import Optional, Union

import numpy as np
from numpy.random import normal
import pandas as pd

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from deepar.dataset.time_series import WindowGenerator
from deepar.model.loss import gaussian_likelihood
from deepar.model import NNModel
from deepar.model.layers import GaussianLayer


logger = logging.getLogger(__name__)


class DeepAR(NNModel):
    def __init__(
        self,
        ts_obj: WindowGenerator,
        epochs=100,
        loss=gaussian_likelihood,
        optimizer="adam",
        with_custom_nn_structure=None,
    ):
        """Init."""

        self.ts_obj = ts_obj
        self.inputs, self.z_sample = None, None
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model = None
        if with_custom_nn_structure:
            self.nn_structure = with_custom_nn_structure
        else:
            self.nn_structure = partial(
                DeepAR.basic_structure,
                n_steps=self.ts_obj.input_width,
                dimensions=len(self.ts_obj.train_df.columns),
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
            4,  # int(4 * (1 + math.pow(math.log(dimensions), 4))),
            return_sequences=True,
            dropout=0.1,
        )(inputs)
        x = Dense(4, activation="relu")(x)  # int(4 * (1 + math.log(dimensions))),
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
        from tensorflow.python.framework.ops import disable_eager_execution

        disable_eager_execution()
        from tensorflow.compat.v1.experimental import output_all_intermediates

        output_all_intermediates(True)

        if not epochs:
            epochs = self.epochs
        callback = callbacks.EarlyStopping(monitor="loss", patience=patience)
        self.keras_model.fit(
            self.ts_obj.train, epochs=epochs, verbose=verbose, callbacks=[callback],
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
        print(model.summary())
        self.keras_model = model

    def instantiate_and_fit(self, do_fit: bool = True, **fit_kwargs):
        """Compile and train model."""
        self.build_model()
        if do_fit:
            self.fit(**fit_kwargs)

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

    def get_sample_prediction(self, sample_df: pd.DataFrame):
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

    import requests

    resp = requests.get(
        "https://github.com/camroach87/gefcom2017data/raw/master/data/gefcom.rda",
        allow_redirects=True,
    )
    open("gefcom.rda", "wb").write(resp.content)
    import pyreadr

    result = pyreadr.read_r("gefcom.rda")
    df = result["gefcom"].pivot(index="ts", columns="zone", values="demand")
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    train_df = pd.DataFrame(data=StandardScaler().fit_transform(df), columns=df.columns)

    ts_window = WindowGenerator(
        input_width=10, label_width=10, shift=10, train_df=train_df
    )
    dp_model = DeepAR(ts_window, epochs=50)
    dp_model.instantiate_and_fit(verbose=1, epochs=500)
