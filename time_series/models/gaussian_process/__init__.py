"""Gaussian Process models."""
import logging
from typing import Optional

import gpflow
from gpflow.utilities import print_summary

from time_series.dataset.time_series import TrainingDataSet
from time_series.models import NNModel


LOGGER = logging.getLogger(__name__)


class GaussianProcess(NNModel):
    """Gaussian Process model based on GPFlow library.

    Data should come in this shape (we'll have to reshape our data to 2D):
    X_train: (instances x variables) -> y_train: (instances x values)
    """
    def __init__(self, data: TrainingDataSet, kernel: gpflow.kernels.Kernel = gpflow.kernels.Matern52(), meanf: Optional[gpflow.mean_functions.MeanFunction] = None):
        self.data = data
        self.kernel = kernel
        print_summary(self.kernel)
        self.meanf = meanf
        self.model: Optional[gpflow.models.BayesianModel] = None
        self.opt = gpflow.optimizers.Scipy()

    def build_model(self):
        """Build model."""
        self.model = gpflow.models.GPR(
            data=(self.data.X_train, self.data.y_train),
            kernel=self.kernel, mean_function=self.meanf
        )
        LOGGER.info(print_summary(self.model))

    def fit(self):
        """Fit the model."""
        _ = self.opt.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=100))
        print_summary(self.model)

    def predict(self, X_test):
        """Return predictions for new data."""
        mean, var = self.model.predict_f(X_test)
        return mean, var
