import abc
from typing import Tuple, cast

import torch

from .. import types
from ._dynamics_model import DynamicsModel
from ._kalman_filter_base import KalmanFilterBase
from ._measurement_models import KalmanFilterMeasurementModel


class ExtendedKalmanFilter(KalmanFilterBase, abc.ABC):
    """Base class for a generic differentiable EKF.

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
        prev_mu = cast(torch.Tensor, self.belief_mu)
        prev_cov = cast(torch.Tensor, self.belief_cov)
        N, state_dim = prev_mu.shape

        # Compute mu_{t+1|t}, covariance, and Jacobian
        pred_mu, dynamics_tril = self.dynamics_model(
            initial_states=prev_mu, controls=controls
        )
        dynamics_covariance = dynamics_tril @ dynamics_tril.transpose(-1, -2)
        dynamics_A_matrix = self.dynamics_model.jacobian(
            states=prev_mu, controls=controls
        )
        assert dynamics_covariance.shape == (N, state_dim, state_dim)
        assert dynamics_A_matrix.shape == (N, state_dim, state_dim)

        # Calculate Sigma_{t+1|t}
        pred_cov = (
            dynamics_A_matrix @ prev_cov @ dynamics_A_matrix.transpose(-1, -2)
            + dynamics_covariance
        )

        return pred_mu, pred_cov

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
        pred_mu, pred_cov = predict_outputs

        # Measurement model forward pass, Jacobian
        measurement_mu = observations
        pred_measurement, measurement_tril = self.measurement_model(states=pred_mu)
        measurement_cov = measurement_tril @ measurement_tril.transpose(-1, -2)
        C_matrix = self.measurement_model.jacobian(states=pred_mu)
        assert measurement_mu.shape == pred_measurement.shape

        # Check shapes
        N, measurement_dim = measurement_mu.shape
        assert measurement_cov.shape == (N, measurement_dim, measurement_dim)
        assert measurement_mu.shape == pred_mu.shape

        # Compute Kalman Gain, innovation
        innovation = measurement_mu - pred_measurement
        innovation_covariance = (
            C_matrix @ pred_cov @ C_matrix.transpose(-1, -2) + measurement_cov
        )
        kalman_gain = (
            pred_cov @ C_matrix.transpose(-1, -2) @ torch.inverse(innovation_covariance)
        )

        # Get mu_{t+1|t+1}, Sigma_{t+1|t+1}
        corrected_mu = pred_mu + (kalman_gain @ innovation[:, :, -1]).unsqueeze(-1)
        assert pred_mu.shape == (N, self.state_dim)

        identity = torch.eye(kalman_gain.shape[-1], device=kalman_gain.device)
        corrected_cov = (identity - kalman_gain) @ pred_cov
        assert corrected_cov.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self.belief_mu = corrected_mu
        self.belief_cov = corrected_cov
