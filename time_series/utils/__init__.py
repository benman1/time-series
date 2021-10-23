"""Utility functions."""
from typing import Sequence

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

from time_series.dataset.time_series import TimeSeries


def set_seed_and_reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def clear_keras_session():
    K.clear_session()


def evaluate_model(tds: TimeSeries, y_predicted: np.ndarray, columns=Sequence[str], first_n: int = 0):
    """Evaluate the model based on the 1step-ahead prediction"""
    print(f"MSE: {mean_squared_error(y_predicted.reshape(-1,), tds.y_test.reshape(-1,)):.4f}")
    print("----------")
    dimensions = len(columns)
    plt.figure(figsize=(12, 18))
    grid = plt.GridSpec(dimensions, 1 if first_n else 2, wspace=0.5, hspace=0.2)

    for i in range(dimensions):
        if len(tds.y_train.shape) == 2:
            pred, y_actual = (
                y_predicted[:first_n, i],
                tds.y_test[:first_n, i]
            )
        else:
            pred, y_actual = (
                y_predicted[:first_n, 1, i],
                tds.y_test[:first_n, 1, i]
            )

        ax = plt.subplot(grid[i, 0])
        plt.plot(pred, 'r+--', label="predicted")
        plt.plot(y_actual, 'bo-.', label="actual")
        ax.set_title(list(columns)[i])
        print(f"{columns[i]}: {round(mean_squared_error(y_actual, pred), 2)}")
    plt.legend()
