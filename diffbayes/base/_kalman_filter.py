import abc

import numpy as np
import torch

import fannypack

from .. import types
from . import DynamicsModel, Filter, KalmanFilterMeasurementModel


class KalmanFilter(Filter, abc.ABC):
    """Base class for a generic differentiable particle filter.
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
                self.state_prev != None and self.states_cov_prev != None
        ), "Kalman filter not initialized!"

        N, state_dim = self.state_prev.shape


        # Dynamics prediction step
        dynamics_pred = self.dynamics_model(self.states_prev)
        states_pred = self.dynamics_model.add_noise(dynamics_pred)
        dynamics_covariance = self.dynamics_model.covariance

        dynamics_A_matrix = self.dynamics_model.jacobian()

        assert dynamics_A_matrix.shape == (N, state_dim, state_dim)

        # Calculating the sigma_t+1|t
        states_sigma_pred = dynamics_A_matrix.bmm(self.states_cov_prev).bmm(dynamics_A_matrix.transpose(-1, -2)) \
                            + dynamics_pred_Q

        measurement_state, measurement_covariance = self.measurement_model(observations)

        # Kalman Gain
        K_update = states_sigma_pred.bmm(torch.inverse(states_sigma_pred + measurement_covariance))

        # Updating
        states_update = torch.unsqueeze(states_pred, -1) \
            + torch.bmm(K_update, torch.unsqueeze((measurement_state - states_pred), -1))
        states_update = states_update.squeeze()
        states_sigma_update = \
            (torch.eye(K_update.shape[-1]).to(K_update.device) - K_update).bmm(states_sigma_pred)

        return states_update, states_sigma_update
