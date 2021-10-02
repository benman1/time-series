import math
import tensorflow as tf


def gaussian_likelihood(sigma):
    """Likelihood as per the paper."""

    def gaussian_loss(y_true, y_pred):
        """Updated from paper.

        See DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
        """
        return tf.reduce_mean(
            tf.math.log(tf.math.sqrt(2 * math.pi))
            + tf.math.log(sigma)
            + tf.math.truediv(
                tf.math.square(y_true - y_pred), 2 * tf.math.square(sigma)
            )
        )

    return gaussian_loss
