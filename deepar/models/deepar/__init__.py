"""Timeseries forecast using a Transformer model.
Based on: Timeseries classification with a Transformer model
By Theodoros Ntakouris, https://github.com/ntakouris
"""
from typing import Optional

import numpy as np
from keras import Model

from deepar.models import NNModel
from tensorflow import keras
from tensorflow.keras import layers


class Transformer(NNModel):
    """Transformer model for time-series.

    The model includes residual connections, layer normalization, and dropout.
    Data come in as (batch size, sequence length, features).
    """
    def __init__(
            self,
            X_train: np.typing.ArrayLike,
            y_train: np.typing.ArrayLike,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = len(np.unique(y_train))
        self.input_shape = X_train.shape[1:]
        self.model: Optional[Model] = None

    def fit(self, **fit_kwargs):
        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        self.model.fit(
            self.X_train,
            self.y_train,
            callbacks=callbacks,
            **fit_kwargs
        )

    def instantiate_and_fit(self, **fit_kwargs):
        """Create model and fit."""
        self.build_model()
        self.fit(**fit_kwargs)

    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
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
            head_size,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=0,
            mlp_dropout=0,
    ):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = Transformer.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        return inputs, outputs

    def build_model(self):
        inputs, outputs = self.nn_structure(
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        self.model = Model(inputs, outputs)
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["sparse_categorical_accuracy"],
        )
        print(self.model.summary())
