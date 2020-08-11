from typing import Tuple, cast

import torch

from .. import types
from ..base._dynamics_model import DynamicsModel
from ..base._kalman_filter_base import KalmanFilterBase
from ..base._kalman_filter_measurement_model import KalmanFilterMeasurementModel


class ExtendedKalmanFilter(KalmanFilterBase):
    """Generic differentiable EKF.

    For building estimators with more complex observation spaces (eg images), see
    `VirtualSensorExtendedKalmanFilter`.
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

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""

        self.measurement_model = measurement_model
        """diffbayes.base.KalmanFilterMeasurementModel: Measurement model."""

    def _predict_step(
        self, *, controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.CovarianceTorch]:
        # Get previous belief
        prev_mean = cast(torch.Tensor, self.belief_mean)
        prev_covariance = cast(torch.Tensor, self.belief_covariance)
        N, state_dim = prev_mean.shape

        # Compute mu_{t+1|t}, covariance, and Jacobian
        pred_mean, dynamics_tril = self.dynamics_model(
            initial_states=prev_mean, controls=controls
        )
        dynamics_covariance = dynamics_tril @ dynamics_tril.transpose(-1, -2)
        dynamics_A_matrix = self.dynamics_model.jacobian(
            initial_states=prev_mean, controls=controls
        )
        assert dynamics_covariance.shape == (N, state_dim, state_dim)
        assert dynamics_A_matrix.shape == (N, state_dim, state_dim)

        # Calculate Sigma_{t+1|t}
        pred_covariance = (
            dynamics_A_matrix @ prev_covariance @ dynamics_A_matrix.transpose(-1, -2)
            + dynamics_covariance
        )

        return pred_mean, pred_covariance

    def _update_step(
        self,
        *,
        predict_outputs: Tuple[types.StatesTorch, types.CovarianceTorch],
        observations: types.ObservationsTorch,
    ) -> None:

        # Extract/validate inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For standard EKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        pred_mean, pred_covariance = predict_outputs

        # Measurement model forward pass, Jacobian
        observations_mean = observations
        pred_observations, observations_tril = self.measurement_model(states=pred_mean)
        observations_covariance = observations_tril @ observations_tril.transpose(
            -1, -2
        )
        C_matrix = self.measurement_model.jacobian(states=pred_mean)
        assert observations_mean.shape == pred_observations.shape

        # Check shapes
        N, observation_dim = observations_mean.shape
        assert observations_covariance.shape == (N, observation_dim, observation_dim)
        assert observations_mean.shape == (N, observation_dim)

        # Compute Kalman Gain, innovation
        innovation = observations_mean - pred_observations
        innovation_covariance = (
            C_matrix @ pred_covariance @ C_matrix.transpose(-1, -2)
            + observations_covariance
        )
        kalman_gain = (
            pred_covariance
            @ C_matrix.transpose(-1, -2)
            @ torch.inverse(innovation_covariance)
        )

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mean = pred_mean + (kalman_gain @ innovation[:, :, None]).squeeze(-1)
        assert corrected_mean.shape == (N, self.state_dim)

        identity = torch.eye(self.state_dim, device=kalman_gain.device)
        corrected_covariance = (identity - kalman_gain @ C_matrix) @ pred_covariance
        assert corrected_covariance.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self.belief_mean = corrected_mean
        self.belief_covariance = corrected_covariance
