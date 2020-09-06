"""Private module; avoid importing from directly.
"""

from typing import Optional, cast

import fannypack
import torch
from overrides import overrides

from .. import types, utils
from ..base import DynamicsModel, KalmanFilterBase, KalmanFilterMeasurementModel


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
        super().__init__(
            dynamics_model=dynamics_model, measurement_model=measurement_model
        )

        # Unscented transform setup
        if sigma_point_strategy is None:
            self._unscented_transform = utils.UnscentedTransform(dim=self.state_dim)
        else:
            self._unscented_transform = utils.UnscentedTransform(
                dim=self.state_dim, sigma_point_strategy=sigma_point_strategy
            )

        # Cache for sigma points; if set, this should always correspond to the current
        # belief distribution
        self._sigma_point_cache: Optional[types.StatesTorch] = None

        # Parameterize posterior uncertainty with lower-triangular covariance root
        self._belief_scale_tril: types.ScaleTrilTorch

    # overrides
    @property
    def belief_covariance(self) -> types.CovarianceTorch:
        return self._belief_scale_tril @ self._belief_scale_tril.transpose(-1, -2)

    # overrides
    @belief_covariance.setter
    def belief_covariance(self, covariance: types.CovarianceTorch):
        self._belief_scale_tril = torch.cholesky(covariance)
        self._belief_covariance = covariance

    @overrides
    def _predict_step(self, *, controls: types.ControlsTorch) -> None:
        """Predict step.
        """
        # See Merwe paper [1] for notation
        x_k_minus_1 = self._belief_mean
        S_k_minus_1 = self._belief_scale_tril
        X_k_minus_1 = self._sigma_point_cache

        N, state_dim = x_k_minus_1.shape

        # Grab sigma points (use cached if available)
        if X_k_minus_1 is None:
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

        # Add sigma dimension back into everything
        X_k_pred = X_k_pred.reshape(X_k_minus_1.shape)
        dynamics_scale_tril = dynamics_scale_tril.reshape(
            (N, sigma_point_count, state_dim, state_dim)
        )

        # Compute predicted distribution
        #
        # Note that we use only the noise term from the first sigma point -- this is
        # slightly different from our standard UKF implementation, which takes a
        # weighted average across all sigma points
        x_k_pred, S_k_pred = self._unscented_transform.compute_distribution_square_root(
            X_k_pred, additive_noise_scale_tril=dynamics_scale_tril[:, 0, :, :]
        )
        assert x_k_pred.shape == (N, state_dim)
        assert S_k_pred.shape == (N, state_dim, state_dim)

        # Update internal state
        self._belief_mean = x_k_pred
        self._belief_scale_tril = S_k_pred
        self._sigma_point_cache = X_k_pred

    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        """Update step.
        """
        # Extract inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For UKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        x_k_pred = self._belief_mean
        S_k_pred = self._belief_scale_tril
        X_k_pred = self._sigma_point_cache
        if X_k_pred is None:
            X_k_pred = self._unscented_transform.select_sigma_points_square_root(
                x_k_pred, S_k_pred
            )

        # Check shapes
        N, sigma_point_count, state_dim = X_k_pred.shape
        observation_dim = self.measurement_model.observation_dim
        assert x_k_pred.shape == (N, state_dim)
        assert S_k_pred.shape == (N, state_dim, state_dim)

        # Propagate sigma points through observation model
        Y_k_pred, measurement_scale_tril = self.measurement_model(
            states=X_k_pred.reshape((-1, state_dim))
        )
        Y_k_pred = Y_k_pred.reshape((N, sigma_point_count, observation_dim))
        measurement_scale_tril = measurement_scale_tril.reshape(
            (N, sigma_point_count, observation_dim, observation_dim)
        )
        assert Y_k_pred.shape == (N, sigma_point_count, observation_dim)
        assert measurement_scale_tril.shape == (
            N,
            sigma_point_count,
            observation_dim,
            observation_dim,
        )

        # Compute observation distribution
        (
            y_k_pred,
            S_y_k_pred,
        ) = self._unscented_transform.compute_distribution_square_root(
            Y_k_pred, additive_noise_scale_tril=measurement_scale_tril[:, 0, :, :]
        )
        assert y_k_pred.shape == (N, observation_dim)
        assert S_y_k_pred.shape == (N, observation_dim, observation_dim)

        # Compute cross-covariance
        X_k_pred_centered = X_k_pred - x_k_pred[:, None, :]
        Y_k_pred_centered = Y_k_pred - y_k_pred[:, None, :]
        P_xy = torch.sum(
            self._unscented_transform.weights_c[None, :, None, None]
            * (X_k_pred_centered[:, :, :, None] @ Y_k_pred_centered[:, :, None, :]),
            dim=1,
        )
        assert P_xy.shape == (N, state_dim, observation_dim)

        # Kalman gain
        # In MATLAB:
        # > K = (P_x_k_y_k / S_y_k_pred.T) / S_y_k
        K = torch.solve(
            torch.solve(P_xy.transpose(-1, -2), S_y_k_pred).solution,
            S_y_k_pred.transpose(-1, -2),
        ).solution.transpose(-1, -2)
        assert K.shape == (N, state_dim, observation_dim)

        # Correct mean
        innovations = observations - y_k_pred
        x_k = x_k_pred + (K @ innovations[:, :, None]).squeeze(-1)

        # Correct uncertainty
        U = K @ S_y_k_pred
        S_k = S_k_pred
        for i in range(U.shape[2]):
            S_k = fannypack.utils.cholupdate(
                S_k, U[:, :, i], weight=torch.tensor(-1.0, device=U.device),
            )

        # Update internal state with corrected beliefs
        self._belief_mean = x_k
        self._belief_scale_tril = S_k
        self._sigma_point_cache = None
