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
        pred_mean, pred_covariance = predict_outputs

        # Use virtual sensor for observation + covariance
        observations_mean, observations_tril = self.virtual_sensor_model(
            observations=observations
        )
        observations_covariance = observations_tril @ observations_tril.transpose(
            -1, -2
        )
        pred_observations = pred_mean

        # Check shapes
        N, observation_dim = observations_mean.shape
        assert observations_covariance.shape == (N, observation_dim, observation_dim)
        assert observations_mean.shape == (N, observation_dim)

        # Compute Kalman Gain, innovation
        innovation = observations_mean - pred_observations
        innovation_covariance = pred_covariance + observations_covariance
        kalman_gain = pred_covariance @ torch.inverse(innovation_covariance)

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mean = pred_mean + (kalman_gain @ innovation[:, :, None]).squeeze(-1)
        assert corrected_mean.shape == (N, self.state_dim)

        identity = torch.eye(self.state_dim, device=kalman_gain.device)
        corrected_covariance = (identity - kalman_gain) @ pred_covariance
        assert corrected_covariance.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self._belief_mean = corrected_mean
        self._belief_covariance = corrected_covariance

    def virtual_sensor_initialize_beliefs(
        self, *, observations: types.ObservationsTorch,
    ):
        """Use virtual sensor model to intialize filter beliefs.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`
        """
        mean, scale_tril = self.virtual_sensor_model(observations)
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        self.initialize_beliefs(mean=mean, covariance=covariance)
