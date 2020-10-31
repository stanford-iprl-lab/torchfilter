"""Private module; avoid importing from directly.
"""
from typing import cast

import fannypack
import torch
from overrides import overrides

from .. import types
from ..base import DynamicsModel, KalmanFilterBase, KalmanFilterMeasurementModel


class ExtendedInformationFilter(KalmanFilterBase):
    """Information form of a Kalman filter; generally equivalent to an EKF but
    internally parameterizes as uncertainties with the inverse covariance matrix.

    For building estimators with more complex observation spaces (eg images), see
    `VirtualSensorExtendedInformationFilter`.
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: KalmanFilterMeasurementModel,
    ):
        super().__init__(
            dynamics_model=dynamics_model, measurement_model=measurement_model
        )

        # Parameterize posterior uncertainty with inverse covariance
        self.information_vector: torch.Tensor
        """torch.Tensor: Information vector of our posterior; shape should be
        `(N, state_dim)."""

        self.information_matrix: torch.Tensor
        """torch.Tensor: Information matrix of our posterior; shape should be
        `(N, state_dim, state_dim)."""

    # # overrides
    # @property
    # def belief_mean(self) -> types.StatesTorch:
    #     """Posterior mean. Shape should be `(N, state_dim)`."""
    #     return self.information_matrix @ self.information_vector
    #
    # # overrides
    # @belief_mean.setter
    # def belief_mean(self, mean: types.StatesTorch):
    #     return self.information_matrix @ self.information_vector

    # overrides
    @property
    def belief_covariance(self) -> types.CovarianceTorch:
        """Posterior covariance. Shape should be `(N, state_dim, state_dim)`."""
        return fannypack.utils.cholesky_inverse(torch.cholesky(self.information_matrix))

    # overrides
    @belief_covariance.setter
    def belief_covariance(self, covariance: types.CovarianceTorch):
        self.information_matrix = fannypack.utils.cholesky_inverse(
            torch.cholesky(covariance)
        )

    @overrides
    def _predict_step(self, *, controls: types.ControlsTorch) -> None:
        # Get previous belief
        prev_mean = self._belief_mean
        prev_covariance = self.belief_covariance
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
        pred_information_matrix = fannypack.utils.cholesky_inverse(
            torch.cholesky(
                dynamics_A_matrix
                @ prev_covariance
                @ dynamics_A_matrix.transpose(-1, -2)
                + dynamics_covariance
            )
        )

        # Update internal state
        self._belief_mean = pred_mean
        self.information_matrix = pred_information_matrix

    @overrides
    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        # Extract/validate inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For standard EKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        pred_mean = self._belief_mean
        pred_information_matrix = self.information_matrix
        pred_information_vector = (
            pred_information_matrix @ pred_mean[:, :, None]
        ).squeeze(-1)

        # Measurement model forward pass, Jacobian
        observations_mean = observations
        pred_observations, observations_tril = self.measurement_model(states=pred_mean)
        observations_information = fannypack.utils.cholesky_inverse(observations_tril)
        C_matrix = self.measurement_model.jacobian(states=pred_mean)
        C_matrix_transpose = C_matrix.transpose(-1, -2)
        assert observations_mean.shape == pred_observations.shape

        # Check shapes
        N, observation_dim = observations_mean.shape
        assert observations_information.shape == (N, observation_dim, observation_dim)
        assert observations_mean.shape == (N, observation_dim)

        # Compute update
        information_vector = pred_information_vector + (
            C_matrix_transpose
            @ observations_information
            @ (
                observations_mean[:, :, None]
                - pred_observations[:, :, None]
                + C_matrix @ pred_mean[:, :, None]
            )
        ).squeeze(-1)
        assert information_vector.shape == (N, self.state_dim)

        information_matrix = (
            pred_information_matrix
            + C_matrix_transpose @ observations_information @ C_matrix
        )
        assert information_matrix.shape == (N, self.state_dim, self.state_dim)

        # Update internal state
        self.information_matrix = information_matrix
        self.information_vector = information_vector
        self._belief_mean = (
            fannypack.utils.cholesky_inverse(torch.cholesky(information_matrix))
            @ information_vector[:, :, None]
        ).squeeze(-1)
