from typing import Optional, Tuple, cast

import torch

import fannypack

from .. import types, utils
from ..base._dynamics_model import DynamicsModel
from ..base._kalman_filter_base import KalmanFilterBase
from ..base._kalman_filter_measurement_model import KalmanFilterMeasurementModel
from ._unscented_kalman_filter import UnscentedKalmanFilter


class SquareRootUnscentedKalmanFilter(KalmanFilterBase):
    """Square-root formulation of UKF.

    From Algorithm 3.1 of Merwe et al [1].

    [1] The square-root unscented Kalman filter for state and parameter-estimation.
    https://ieeexplore.ieee.org/document/940586/
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: KalmanFilterMeasurementModel,
        sigma_point_strategy: Optional[utils.SigmaPointStrategy] = None,
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, KalmanFilterMeasurementModel)

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Unscented transform setup
        if sigma_point_strategy is None:
            self._unscented_transform = utils.UnscentedTransform(dim=state_dim)
        else:
            self._unscented_transform = utils.UnscentedTransform(
                dim=state_dim, sigma_point_strategy=sigma_point_strategy
            )

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""

        self.measurement_model = measurement_model
        """diffbayes.base.KalmanFilterMeasurementModel: Measurement model."""

        # Parameterize posterior uncertainty with lower-triangular covariance root
        self._belief_scale_tril: types.ScaleTrilTorch

    @property
    def belief_covariance(self) -> types.CovarianceTorch:
        return self._belief_scale_tril @ self._belief_scale_tril.transpose(-1, -2)

    @belief_covariance.setter
    def belief_covariance(self, covariance: types.CovarianceTorch):
        self._belief_scale_tril = torch.cholesky(covariance)
        self._belief_covariance = covariance

    def _predict_step(
        self, *, controls: types.ControlsTorch,
    ) -> Tuple[
        types.StatesTorch,
        types.ScaleTrilTorch,
        types.StatesTorch,
        types.ObservationsNoDictTorch,
        types.ScaleTrilTorch,
    ]:
        """Predict step.

        Outputs...
        - Predicted mean
        - Predicted covariance (square root)
        - State sigma points
        - Observation sigma points
        - Measurement model covariances (square root)
        """
        # self._belief_covariance = (
        #     self._belief_scale_tril @ self._belief_scale_tril.transpose(-1, -2)
        # )
        self._weighted_covariance = lambda sigma_trils: UnscentedKalmanFilter._weighted_covariance(
            self, sigma_trils
        )

        x_k_minus_1 = self.belief_mean
        S_k_minus_1 = self._belief_scale_tril
        N, state_dim = x_k_minus_1.shape
        observation_dim = self.measurement_model.observation_dim

        # Grab sigma points
        X_k_minus_1 = self._unscented_transform.select_sigma_points_square_root(
            x_k_minus_1, S_k_minus_1
        )
        sigma_point_count = state_dim * 2 + 1
        assert X_k_minus_1.shape == (N, sigma_point_count, state_dim)

        # Flatten sigma points and propagate through dynamics, then measurement models
        X_k_pred, dynamics_scale_tril = self.dynamics_model(
            initial_states=X_k_minus_1.reshape((-1, state_dim)),
            controls=fannypack.utils.SliceWrapper(controls).map(
                lambda tensor: torch.repeat_interleave(
                    tensor, repeats=sigma_point_count, dim=0
                )
            ),
        )
        Y_k_pred, measurement_scale_tril = self.measurement_model(states=X_k_pred)

        # Add sigma dimension back into everything
        X_k_pred = X_k_pred.reshape(X_k_minus_1.shape)
        dynamics_scale_tril = dynamics_scale_tril.reshape(
            (N, sigma_point_count, state_dim, state_dim)
        )
        Y_k_pred = Y_k_pred.reshape((N, sigma_point_count, observation_dim))
        measurement_scale_tril = measurement_scale_tril.reshape(
            (N, sigma_point_count, observation_dim, observation_dim)
        )

        # Compute predicted distribution
        #
        # Note that we use only the noise term from the first sigma point -- this is
        # slightly different from our standard UKF implementation, which takes a
        # weighted average across all sigma points
        (
            x_k_pred,
            S_x_pred_k,
        ) = self._unscented_transform.compute_distribution_square_root(
            X_k_pred, additive_noise_scale_tril=dynamics_scale_tril[:, 0, :, :],
        )
        assert x_k_pred.shape == (N, state_dim)

        return (
            x_k_pred,
            S_x_pred_k,
            X_k_pred,
            Y_k_pred,
            measurement_scale_tril[:, 0, :, :],
        )

    def _update_step(
        self,
        *,
        predict_outputs: Tuple[
            types.StatesTorch,
            types.ScaleTrilTorch,
            types.StatesTorch,
            types.ObservationsNoDictTorch,
            types.ScaleTrilTorch,
        ],
        observations: types.ObservationsTorch,
    ) -> None:
        # Extract inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For UKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        (
            x_k_pred,
            S_x_pred_k,
            X_k_pred,
            Y_k_pred,
            measurement_scale_tril,
        ) = predict_outputs

        # Check shapes
        N, sigma_point_count, state_dim = X_k_pred.shape
        observation_dim = self.measurement_model.observation_dim
        assert x_k_pred.shape == (N, state_dim)
        assert S_x_pred_k.shape == (N, state_dim, state_dim)
        assert Y_k_pred.shape == (N, sigma_point_count, observation_dim,)

        # Compute observation distribution
        (
            y_k_pred,
            S_y_pred_k,
        ) = self._unscented_transform.compute_distribution_square_root(
            Y_k_pred, additive_noise_scale_tril=measurement_scale_tril
        )
        assert y_k_pred.shape == (N, observation_dim)
        assert S_y_pred_k.shape == (N, observation_dim, observation_dim,)

        # Compute cross-covariance
        centered_sigma_points = X_k_pred - x_k_pred[:, None, :]
        centered_sigma_observations = Y_k_pred - y_k_pred[:, None, :]
        P_x_k_y_k = torch.sum(
            self._unscented_transform.weights_c[None, :, None, None]
            * (
                centered_sigma_points[:, :, :, None]
                @ centered_sigma_observations[:, :, None, :]
            ),
            dim=1,
        )
        assert P_x_k_y_k.shape == (N, state_dim, observation_dim,)

        # Kalman gain, innovation
        # In MATLAB:
        # > K = (P_x_k_y_k / S_y_pred_k.T) / S_y_k
        K = torch.solve(
            torch.solve(P_x_k_y_k.transpose(-1, -2), S_y_pred_k).solution,
            S_y_pred_k.transpose(-1, -2),
        ).solution.transpose(-1, -2)
        assert K.shape == (N, state_dim, observation_dim)

        # Correct mean
        innovations = observations - y_k_pred
        x_k = x_k_pred + (K @ innovations[:, :, None]).squeeze(-1)

        # Correct uncertainty
        U = K @ S_y_pred_k
        S_x_k = S_x_pred_k
        for i in range(U.shape[2]):
            S_x_k = fannypack.utils.cholupdate(
                S_x_k, U[:, :, i], weight=torch.tensor(-1.0, device=U.device),
            )

        # Update internal state with corrected beliefs
        self._belief_mean = x_k
        self._belief_scale_tril = S_x_k
