"""Template for models."""
from abc import ABC
import tensorflow as tf


class NNModel(ABC):
    """Model class."""

    metrics = ["mean_absolute_percentage_error", "mae", "mse"]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True
        )
    ]

    def __init__(self):
        super().__init__()

    def net_structure(self, **kwargs):
        pass

    def instantiate_and_fit(self, **kwargs):
        pass

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model

        return load_model(filepath, custom_objects, compile)

    @property
    def model(self):
        raise AttributeError("Not implemented!")
