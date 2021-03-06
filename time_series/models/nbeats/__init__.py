"""Keras implementation of N-BEATS.

Based on Philippe Rémy's implementation at https://github.com/philipperemy/n-beats.
Paper: NBEATS: Neural basis expansion analysis for interpretable time series forecasting
"""
import logging
from typing import Dict, Optional

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
from tensorflow.keras.models import Model

from time_series.dataset.time_series import TrainingDataSet
from time_series.models import NNModel


LOGGER = logging.getLogger(__name__)

GENERIC_BLOCK = "generic"
TREND_BLOCK = "trend"
SEASONALITY_BLOCK = "seasonality"

_BACKCAST = "backcast"
_FORECAST = "forecast"


def linear_space(backcast_length, forecast_length, is_forecast=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    return (
        ls[backcast_length:]
        if is_forecast
        else K.abs(K.reverse(ls[:backcast_length], axes=0))
    )


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))


class NBeatsNet(NNModel):
    """NBeats model with exogenous variables.

    Data come in as (num_samples, time_steps, input_dim).

    We could be moving a window generator here:
    self.ts_obj = WindowGenerator(input_width=10, label_width=10, shift=8, train_df=df)
    """

    cast_type: str = _FORECAST

    def __init__(
        self,
        data: TrainingDataSet,
        backcast_length=10,
        stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
        nb_blocks_per_stack=3,
        thetas_dim=(4, 8),
        share_weights_in_stack=False,
        hidden_layer_units=256,
        nb_harmonics=None,
    ):
        self.data = data
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.input_shape = (self.backcast_length, self.data.dimensions)
        self.exo_shape = (self.backcast_length, self.data.exo_dim)
        self.output_shape = (self.data.n_steps, self.data.dimensions)
        self.weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)
        self.models: Optional[Dict[str, Model]] = None

    def net_structure(self):
        """Build the network structure."""
        x = Input(shape=self.input_shape, name="input_variable")
        x_ = {}
        for k in range(self.data.dimensions):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name="exos_variables")
            for k in range(self.data.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(
                    x_, e_, stack_id, block_id, stack_type, nb_poly
                )
                for k in range(self.data.dimensions):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.data.dimensions):
            y_[k] = Reshape(target_shape=(self.data.n_steps, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.data.dimensions > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.data.dimensions)])
            x_ = Concatenate()([x_[ll] for ll in range(self.data.dimensions)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.has_exog():
            n_beats_forecast = Model([x, e], y_, name=_FORECAST)
            n_beats_backcast = Model([x, e], x_, name=_BACKCAST)
        else:
            n_beats_forecast = Model(x, y_, name=_FORECAST)
            n_beats_backcast = Model(x, x_, name=_BACKCAST)
        return n_beats_forecast, n_beats_backcast

    def build_model(self):
        """Build the models."""
        n_beats_forecast, n_beats_backcast = self.net_structure()
        self.models = {
            model.name: model for model in [n_beats_backcast, n_beats_forecast]
        }
        self.models[_FORECAST].compile(loss="mae", optimizer="adam")
        LOGGER.info(self.models[_FORECAST].summary())

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.data.exo_dim > 0

    def _restore(self, layer_with_weights, stack_id):
        """Mechanism to restore weights when block share the same weights.

        This is only useful when share_weights_in_stack=True.
        """
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split("/")[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        """Register weights.

        This is useful when share_weights_in_stack=True.
        """

        def reg(layer):
            return self._restore(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return "/".join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(self.units, activation="relu", name=n("d1")))
        d2 = reg(Dense(self.units, activation="relu", name=n("d2")))
        d3 = reg(Dense(self.units, activation="relu", name=n("d3")))
        d4 = reg(Dense(self.units, activation="relu", name=n("d4")))
        if stack_type == "generic":
            theta_b = reg(
                Dense(nb_poly, activation="linear", use_bias=False, name=n("theta_b"))
            )
            theta_f = reg(
                Dense(nb_poly, activation="linear", use_bias=False, name=n("theta_f"))
            )
            backcast = reg(
                Dense(self.backcast_length, activation="linear", name=n("backcast"))
            )
            forecast = reg(
                Dense(self.data.n_steps, activation="linear", name=n("forecast"))
            )
        elif stack_type == "trend":
            theta_f = theta_b = reg(
                Dense(nb_poly, activation="linear", use_bias=False, name=n("theta_f_b"))
            )
            backcast = Lambda(
                trend_model,
                arguments={
                    "is_forecast": False,
                    "backcast_length": self.backcast_length,
                    "forecast_length": self.data.n_steps,
                },
            )
            forecast = Lambda(
                trend_model,
                arguments={
                    "is_forecast": True,
                    "backcast_length": self.backcast_length,
                    "forecast_length": self.data.n_steps,
                },
            )
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(
                    Dense(
                        self.nb_harmonics,
                        activation="linear",
                        use_bias=False,
                        name=n("theta_b"),
                    )
                )
            else:
                theta_b = reg(
                    Dense(
                        self.data.n_steps,
                        activation="linear",
                        use_bias=False,
                        name=n("theta_b"),
                    )
                )
            theta_f = reg(
                Dense(
                    self.data.n_steps,
                    activation="linear",
                    use_bias=False,
                    name=n("theta_f"),
                )
            )
            backcast = Lambda(
                seasonality_model,
                arguments={
                    "is_forecast": False,
                    "backcast_length": self.backcast_length,
                    "forecast_length": self.data.n_steps,
                },
            )
            forecast = Lambda(
                seasonality_model,
                arguments={
                    "is_forecast": True,
                    "backcast_length": self.backcast_length,
                    "forecast_length": self.data.n_steps,
                },
            )
        for k in range(self.data.dimensions):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def fit(self, **fit_kwargs):
        """Fit model."""
        self.models[_FORECAST].fit(
            self.data.X_train, self.data.y_train, callbacks=self.callbacks, **fit_kwargs
        )

    def instantiate_and_fit(self, **fit_kwargs):
        self.build_model()
        LOGGER.info("Model built!")
        self.fit(**fit_kwargs)

    @property
    def model(self):
        """Get the forecast model."""
        return self.models[_FORECAST]
