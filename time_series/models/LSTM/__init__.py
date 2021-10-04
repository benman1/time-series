"""Recurrent neural network."""
import tensorflow as tf

from time_series.dataset.time_series import TrainingDataSet
from time_series.models.transformer import Transformer


class LSTM(Transformer):
    """Forecasting with an LSTM."""
    def __init__(self, data: TrainingDataSet):
        super().__init__(data)

    def build_model(self):
        """Build model."""
        inputs = tf.keras.Input(shape=self.data.input_shape)
        x = tf.keras.layers.LSTM(units=100, return_sequences=True)(inputs)
        x = tf.keras.layers.Dense(self.data.n_steps)(x)
        self.model = tf.keras.Model(inputs, x)
        self.model.compile(
            loss="mse" if self.regression else "sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=self.metrics,
        )
        print(self.model.summary())
