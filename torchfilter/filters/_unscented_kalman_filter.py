"""Private module; avoid importing from directly.
"""

from typing import Optional, cast

import fannypack
import torch
from overrides import overrides

from .. import types, utils
from ..base import DynamicsModel, KalmanFilterBase, KalmanFilterMeasurementModel


class UnscentedKalmanFilter(KalmanFilterBase):
    """Standard UKF.

    From Algorithm 2.1 of Merwe et al. [1]. For working with heteroscedastic noise
    models, we use the weighting approach described in [2].

    [1] The square-root unscented Kalman filter for state and parameter-estimation.
    https://ieeexplore.ieee.org/document/940586/
    [2] How to Train Your Differentiable Filter
    https://al.is.tuebingen.mpg.de/uploads_file/attachment/attachment/617/2020_RSS_WS_alina.pdf
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

    @overrides
    def _predict_step(self, *, controls: types.ControlsTorch) -> None:
        """Predict step."""
        # See Merwe paper [1] for notation
        x_k_minus_1 = self._belief_mean
        P_k_minus_1 = self._belief_covariance
        X_k_minus_1 = self._sigma_point_cache

        N, state_dim = x_k_minus_1.shape

        # Grab sigma points (use cached if available)
        if X_k_minus_1 is None:
            X_k_minus_1 = self._unscented_transform.select_sigma_points(
                x_k_minus_1, P_k_minus_1
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
        x_k_pred, P_k_pred = self._unscented_transform.compute_distribution(X_k_pred)
        assert x_k_pred.shape == (N, state_dim)
        assert P_k_pred.shape == (N, state_dim, state_dim)

        # Compute weighted covariances (see helper docstring for explanation)
        dynamics_covariance = self._weighted_covariance(dynamics_scale_tril)
        assert dynamics_covariance.shape == (N, state_dim, state_dim)

        # Add dynamics uncertainty
        P_k_pred = P_k_pred + dynamics_covariance

        # Update internal state
        self._belief_mean = x_k_pred
        self._belief_covariance = P_k_pred
        self._sigma_point_cache = X_k_pred

    @overrides
    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        """Update step."""
        # Extract inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For UKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        x_k_pred = self._belief_mean
        P_k_pred = self._belief_covariance
        X_k_pred = self._sigma_point_cache
        if X_k_pred is None:
            X_k_pred = self._unscented_transform.select_sigma_points(x_k_pred, P_k_pred)

        # Check shapes
        N, sigma_point_count, state_dim = X_k_pred.shape
        observation_dim = self.measurement_model.observation_dim
        assert x_k_pred.shape == (N, state_dim)
        assert P_k_pred.shape == (N, state_dim, state_dim)

        # Propagate sigma points through observation model
        Y_k_pred, measurement_scale_tril = self.measurement_model(
            states=X_k_pred.reshape((-1, state_dim))
        )
        Y_k_pred = Y_k_pred.reshape((N, sigma_point_count, observation_dim))
        measurement_scale_tril = measurement_scale_tril.reshape(
            (N, sigma_point_count, observation_dim, observation_dim)
        )
        measurement_covariance = self._weighted_covariance(measurement_scale_tril)
        assert Y_k_pred.shape == (N, sigma_point_count, observation_dim)
        assert measurement_covariance.shape == (N, observation_dim, observation_dim)

        # Compute observation distribution
        y_k_pred, P_y_k_pred = self._unscented_transform.compute_distribution(Y_k_pred)
        P_y_k_pred = P_y_k_pred + measurement_covariance
        assert y_k_pred.shape == (N, observation_dim)
        assert P_y_k_pred.shape == (N, observation_dim, observation_dim)

        # Compute cross-covariance
        X_k_pred_centered = X_k_pred - x_k_pred[:, None, :]
        Y_k_pred_centered = Y_k_pred - y_k_pred[:, None, :]
        P_xy = torch.sum(
            self._unscented_transform.weights_c[None, :, None, None]
            * (X_k_pred_centered[:, :, :, None] @ Y_k_pred_centered[:, :, None, :]),
            dim=1,
        )
        assert P_xy.shape == (N, state_dim, observation_dim)

        # Kalman gain, innovation
        K = P_xy @ torch.inverse(P_y_k_pred)
        assert K.shape == (N, state_dim, observation_dim)

        # Correct mean
        innovations = observations - y_k_pred
        x_k = x_k_pred + (K @ innovations[:, :, None]).squeeze(-1)

        # Correct covariance
        P_k = P_k_pred - K @ P_y_k_pred @ K.transpose(-1, -2)

        # Update internal state with corrected beliefs
        self._belief_mean = x_k
        self._belief_covariance = P_k
        self._sigma_point_cache = None

    def _weighted_covariance(
        self, sigma_trils: types.ScaleTrilTorch
    ) -> types.CovarianceTorch:
        """For heteroscedastic covariances, we apply the weighted average approach
        described by Kloss et al:
        https://homes.cs.washington.edu/~barun/files/workshops/rss2020_sarl/submissions/7_differentiablefilter.pdf

        (note that the mean weights are used because they sum to 1)
        """
        N, sigma_point_count, dim, dim_alt = sigma_trils.shape
        assert dim == dim_alt

        if sigma_trils.stride()[:2] == (0, 0):
            # All covariances identical => we can do less math
            output_covariance = sigma_trils[0, 0] @ sigma_trils[0, 0].transpose(-1, -2)
            assert output_covariance.shape == (dim, dim), output_covariance.shape
            output_covariance = output_covariance[None, :, :].expand((N, dim, dim))
        else:
            # Otherwise, compute weighted covariance
            pred_sigma_tril = sigma_trils.reshape((N, sigma_point_count, dim, dim))
            pred_sigma_covariance = pred_sigma_tril @ pred_sigma_tril.transpose(-1, -2)
            output_covariance = torch.sum(
                self._unscented_transform.weights_m[None, :, None, None]
                * pred_sigma_covariance,
                dim=1,
            )
        return output_covariance
