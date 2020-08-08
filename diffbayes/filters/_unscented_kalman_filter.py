from typing import Dict, Tuple, cast

import torch

from .. import types, utils
from ..base._dynamics_model import DynamicsModel
from ..base._measurement_models import KalmanFilterMeasurementModel
from ..base._kalman_filter_base import KalmanFilterBase


class UnscentedKalmanFilter(KalmanFilterBase):
    """Generic (naive) UKF.
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: KalmanFilterMeasurementModel,
        unscented_transform_params: Dict[str, float],
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, KalmanFilterMeasurementModel)

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Unscented transform setup
        self._unscented_transform = utils.UnscentedTransform(
            dim=state_dim, **unscented_transform_params
        )

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""

        self.measurement_model = measurement_model
        """diffbayes.base.KalmanFilterMeasurementModel: Measurement model."""

    def _predict_step(
        self, *, controls: types.ControlsTorch,
    ) -> Tuple[
        types.StatesTorch,
        types.CovarianceTorch,
        types.StatesTorch,
        types.ObservationsNoDictTorch,
        types.CovarianceTorch,
    ]:
        """Predict step.

        Outputs...
        - Predicted mean
        - Predicted covariance
        - State sigma points
        - Observation sigma points
        - Measurement model covariances
        """
        prev_mu = cast(torch.Tensor, self.belief_mu)
        prev_cov = cast(torch.Tensor, self.belief_cov)
        N, state_dim = prev_mu.shape
        observation_dim = self.measurement_model.observation_dim

        # Grab sigma points
        sigma_points = self._unscented_transform.select_sigma_points(prev_mu, prev_cov)
        sigma_point_count = state_dim * 2 + 1
        assert sigma_points.shape == (N, sigma_point_count, state_dim)

        # Propagate through dynamics, then measurement models
        pred_sigma_points, pred_sigma_tril = self.dynamics_model(
            initial_states=sigma_points.reshape((N, -1)), controls=controls
        )
        pred_sigma_observations, pred_sigma_observations_tril = self.measurement_model(
            pred_sigma_points
        ).reshape((N, sigma_point_count, self.measurement_model.observation_dim))
        pred_sigma_points = pred_sigma_points.reshape(sigma_points.shape)

        # Compute predicted distribution
        pred_mu, pred_cov = self._unscented_transform.compute_distribution(
            pred_sigma_points
        )
        assert pred_mu.shape == (N, state_dim)
        assert pred_cov.shape == (N, state_dim, state_dim)

        # Compute weighted covariances (see helper docstring for explanation)
        dynamics_cov = self._weighted_covariances(
            pred_sigma_tril.reshape((N, sigma_point_count, state_dim, state_dim))
        )
        assert dynamics_cov.shape == (N, state_dim, state_dim)
        measurement_cov = self._weighted_covariances(
            pred_sigma_observations_tril.reshape(
                (N, sigma_point_count, observation_dim, observation_dim)
            )
        )
        assert measurement_cov.shape == (N, observation_dim, observation_dim)

        # Add dynamics uncertainty
        pred_cov = pred_cov + dynamics_cov

        return (
            pred_mu,
            pred_cov,
            pred_sigma_points,
            pred_sigma_observations,
            measurement_cov,
        )

    def _update_step(
        self,
        *,
        predict_outputs: Tuple[
            types.StatesTorch,
            types.CovarianceTorch,
            types.StatesTorch,
            types.ObservationsNoDictTorch,
            types.CovarianceTorch,
        ],
        observations: types.ObservationsTorch,
    ) -> None:
        (
            pred_mu,
            pred_cov,
            pred_sigma_points,
            pred_sigma_observations,
            measurement_cov,
        ) = predict_outputs

        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For UKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)

        # Check shapes
        N, sigma_point_count, state_dim = pred_sigma_points
        observation_dim = self.measurement_model.observation_dim
        assert pred_mu.shape == (N, state_dim)
        assert pred_cov.shape == (N, state_dim, state_dim)
        assert pred_sigma_observations.shape == (N, sigma_point_count, observation_dim,)

        # Compute observation distribution
        (
            pred_observations,
            pred_observations_cov,
        ) = self._unscented_transform.compute_distribution(pred_sigma_observations)
        assert pred_observations.shape == (N, observation_dim)
        assert pred_observations_cov.shape == (N, observation_dim, observation_dim)

        # Add measurement model uncertainty
        pred_observations_cov = pred_observations_cov + measurement_cov

        # Compute cross-covariance
        centered_sigma_points = pred_sigma_points - pred_mu[:, None, :]
        centered_sigma_observations = (
            pred_sigma_observations - pred_observations[:, None, :]
        )
        cross_covariance = torch.sum(
            self._unscented_transform.weights_c
            * (centered_sigma_points @ centered_sigma_observations.transpose(-1, -2)),
            dim=1,
        )
        assert cross_covariance.shape == (N, state_dim, observation_dim)

        # Kalman gain, innovation
        K = cross_covariance @ torch.inverse(pred_observations_cov)
        assert K.shape == (N, state_dim, observation_dim)

        innovations = observations - pred_observations

        # Update internal state with corrected beliefs
        self.belief_mu = pred_mu + K @ innovations
        self.belief_cov = pred_cov - K @ pred_observations_cov @ K.transpose(-1, -2)

    def _weighted_covariances(self, sigma_trils: torch.Tensor):
        """For heteroscedastic covariances, we apply the weighted average approach
        described by Kloss et al:
        https://homes.cs.washington.edu/~barun/files/workshops/rss2020_sarl/submissions/7_differentiablefilter.pdf

        (note that the mean weights are used because they sum to 1)
        """
        N, sigma_point_count, dim, _ = sigma_trils.shape

        if sigma_trils.stride()[:2] == (0, 0):
            # All covariances identical => we can do less math
            output_cov = sigma_trils[0] @ sigma_trils[0].transpose(-1, -2)
            assert output_cov.shape == (dim, dim)
            output_cov = output_cov[None, :, :].expand((N, dim, dim))
        else:
            pred_sigma_tril = sigma_trils.reshape((N, sigma_point_count, dim, dim))
            pred_sigma_cov = pred_sigma_tril @ pred_sigma_tril.transpose(-1, -2)
            output_cov = torch.sum(
                self._unscented_transform.weights_m[None, :, None, None]
                * pred_sigma_cov,
                dim=1,
            )
        return output_cov
