"""Time-series forecast using a Transformer model.
Based on: Timeseries classification with a Transformer model
By Theodoros Ntakouris, https://github.com/ntakouris
"""
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras import Model

from time_series.dataset.time_series import TrainingDataSet
from time_series.models import NNModel
from tensorflow.keras import layers


class Transformer(NNModel):
    """Transformer model for time-series.

    The model includes residual connections, layer normalization, and dropout.
    Data come in as (batch size, sequence length, features).
    """

    def __init__(
        self, data: TrainingDataSet,
        regression: bool = True
    ):
        self.data = data
        self.model: Optional[Model] = None
        self.regression = regression

    def fit(self, **fit_kwargs):
        self.model.fit(
            self.data.X_train, self.data.y_train, callbacks=self.callbacks, **fit_kwargs
        )

    def instantiate_and_fit(self, **fit_kwargs):
        """Create model and fit."""
        self.build_model()
        self.fit(**fit_kwargs)

    @staticmethod
    def transformer_encoder(
        inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0.0
    ):
        """Encoder: Attention and Normalization and Feed-Forward."""
        # 1. Attention and Normalization:
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # 2. Feed Forward Part:
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def nn_structure(
        self,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        num_transformer_blocks: int,
        mlp_units: Sequence[int],
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        inputs = tf.keras.Input(shape=self.data.input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = Transformer.transformer_encoder(
                x, head_size, num_heads, ff_dim, dropout
            )

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs_d = layers.Dense(
            self.data.dimensions * self.data.n_steps if self.regression else self.data.n_classes,
            activation="softmax"
        )(x)
        outputs = tf.reshape(outputs_d, (-1, self.data.dimensions, self.data.lag))
        return inputs, outputs

    def build_model(self):
        inputs, outputs = self.nn_structure(
            head_size=256,
            num_heads=1,
            ff_dim=4,
            num_transformer_blocks=1,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        self.model = Model(inputs, outputs)
        self.model.compile(
            loss="mse" if self.regression else "sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=self.metrics,
        )
        print(self.model.summary())
