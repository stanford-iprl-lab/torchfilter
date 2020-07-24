import abc

import numpy as np
import torch

import fannypack

from .. import types
from . import DynamicsModel, Filter, KalmanFilterMeasurementModel


class KalmanFilter(Filter, abc.ABC):
    """Base class for a generic differentiable extended kalman filter.
        We assume measurement model gives us the state directly.
    """

    def __init__(
            self,
            *,
            dynamics_model: DynamicsModel,
            measurement_model: KalmanFilterMeasurementModel,
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, KalmanFilterMeasurementModel)
        assert dynamics_model.state_dim == measurement_model.state_dim

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""
        self.measurement_model = measurement_model
        """diffbayes.base.KalmanFilterMeasurementModel: Observation model."""

        self.states_prev = None
        self.states_covariance_prev = None

    def forward(
            self, *, observations: types.ObservationsTorch,
            controls: types.ControlsTorch,
    ) -> (types.StatesTorch, types.CovarianceTorch):
        """Kalman filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
            torch.Tensor: Predicted state covariance for each batch element. Shape should
            be `(N, state_dim, state_dim).`
        """
        assert (
                self.states_prev is not None and self.states_covariance_prev is not None
        ), "Kalman filter not initialized!"

        N, state_dim = self.states_prev.shape

        # Dynamics prediction step
        predicted_states, dynamics_tril = self.dynamics_model(initial_states=self.states_prev,
                                                              controls=controls)
        dynamics_noise = dynamics_tril.bmm(dynamics_tril.transpose(-1, -2))
        dynamics_A_matrix = self.dynamics_model.jacobian(self.states_prev,
                                                         controls,
                                                         self.dynamics_model)
        assert dynamics_A_matrix.shape == (N, state_dim, state_dim)
        # Calculating the sigma_t+1|t
        predicted_covariances = dynamics_A_matrix.bmm(self.states_covariance_prev).bmm(
                                dynamics_A_matrix.transpose(-1, -2)) + dynamics_noise

        measurement_prediction, measurement_covariance = self.measurement_model(observations=observations)

        # Kalman Gain
        kalman_update = predicted_covariances.bmm(torch.inverse(predicted_covariances +
                                                                measurement_covariance))

        # Updating
        states_estimate = torch.unsqueeze(predicted_states, -1) \
            + torch.bmm(kalman_update, torch.unsqueeze((measurement_prediction - predicted_states), -1))
        states_estimate = states_estimate.squeeze()
        states_covariance_estimate = (torch.eye(kalman_update.shape[-1]).to(
            kalman_update.device) - kalman_update).bmm(
            predicted_covariances)

        self.states_prev = states_estimate
        self.states_covariance_prev = states_covariance_estimate

        return states_estimate

    @property
    def state_covariance_estimate(self):
        return self._states_covariance_prev

    def initialize_beliefs(self, *, mean: torch.Tensor, covariance: torch.Tensor):
        """Set kalman state prediction and state covariance to mean and covariance.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        self.states_prev = mean
        self.states_covariance_prev = covariance

    def measurement_initialize_beliefs(self, *, observations: types.ObservationsTorch,):
        """Use measurement model to intialize belief.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`
        """
        measurement_prediction, measurement_covariance = self.measurement_model(observations)
        self.initialize_beliefs(mean=measurement_prediction, covariance=measurement_covariance)