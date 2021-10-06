"""Recurrent neural network."""
from typing import Sequence

import tensorflow as tf

from time_series.dataset.time_series import TrainingDataSet
from time_series.models.transformer import Transformer


class LSTM(Transformer):
    """Forecasting with an LSTM."""
    def __init__(self, data: TrainingDataSet, lstm_units: Sequence[int] = (100,)):
        super().__init__(data)
        self.lstm_units = lstm_units

    def recurrent_layers(self, inputs):
        x = inputs
        for i, dim in enumerate(self.lstm_units):
            x = tf.keras.layers.LSTM(
                units=dim,
                return_sequences=True
            )(x)
        x = tf.keras.layers.Dense(self.data.dimensions)(x)
        return x

    def build_model(self):
        """Build model."""
        inputs = tf.keras.Input(shape=self.data.input_shape)
        lstm_output = self.recurrent_layers(inputs)
        self.model = tf.keras.Model(inputs, lstm_output)
        self.model.compile(
            loss="mse" if self.regression else "sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=self.metrics,
        )
        print(self.model.summary())
