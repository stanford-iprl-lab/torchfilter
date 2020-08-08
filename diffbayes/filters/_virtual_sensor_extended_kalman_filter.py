import abc
from typing import Tuple, cast

import torch

from .. import types
from ..base._dynamics_model import DynamicsModel
from ..base._kalman_filter_base import KalmanFilterBase
from ..base._virtual_sensor_model import VirtualSensorModel
from ._extended_kalman_filter import ExtendedKalmanFilter


class VirtualSensorExtendedKalmanFilter(KalmanFilterBase):
    """Generic, BackpropKF-style EKF with a virtual sensor model for mapping raw
    observations to predicted states.

    Assumes measurement model is identity.
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        virtual_sensor_model: VirtualSensorModel,
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(virtual_sensor_model, VirtualSensorModel)

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""
        self.virtual_sensor_model = virtual_sensor_model
        """diffbayes.base.VirtualSensorModel: Virtual sensor model."""

    def _predict_step(
        self, *, controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.CovarianceTorch]:
        # Same as normal EKF
        return ExtendedKalmanFilter._predict_step(
            cast(ExtendedKalmanFilter, self), controls=controls,
        )

    def _update_step(
        self,
        *,
        predict_outputs: Tuple[types.StatesTorch, types.CovarianceTorch],
        observations: types.ObservationsTorch,
    ) -> None:

        # Extract inputs
        pred_mu, pred_cov = predict_outputs

        # Use virtual sensor for observation + covariance
        observations_mu, observations_tril = self.virtual_sensor_model(
            observations=observations
        )
        observations_cov = observations_tril @ observations_tril.transpose(-1, -2)
        pred_observations = pred_mu

        # Check shapes
        N, observation_dim = observations_mu.shape
        assert observations_cov.shape == (N, observation_dim, observation_dim)
        assert observations_mu.shape == pred_mu.shape

        # Compute Kalman Gain, innovation
        innovation = observations_mu - pred_observations
        innovation_covariance = pred_cov + observations_cov
        kalman_gain = pred_cov @ torch.inverse(innovation_covariance)

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mu = pred_mu + (kalman_gain @ innovation[:, :, -1]).unsqueeze(-1)
        assert pred_mu.shape == (N, self.state_dim)

        identity = torch.eye(kalman_gain.shape[-1], device=kalman_gain.device)
        corrected_cov = (identity - kalman_gain) @ pred_cov
        assert corrected_cov.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self.belief_mu = corrected_mu
        self.belief_cov = corrected_cov

    def virtual_sensor_initialize_beliefs(
        self, *, observations: types.ObservationsTorch,
    ):
        """Use virtual sensor model to intialize filter beliefs.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`
        """
        mean, covariance = self.virtual_sensor_model(observations)
        self.initialize_beliefs(mean=mean, covariance=covariance)
