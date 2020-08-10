from typing import Dict, Tuple, cast

import torch

from .. import types, utils
from ..base._dynamics_model import DynamicsModel
from ..base._kalman_filter_base import KalmanFilterBase
from ..base._measurement_models import KalmanFilterMeasurementModel


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
        prev_mean = cast(torch.Tensor, self.belief_mean)
        prev_covariance = cast(torch.Tensor, self.belief_covariance)
        N, state_dim = prev_mean.shape
        observation_dim = self.measurement_model.observation_dim

        # Grab sigma points
        sigma_points = self._unscented_transform.select_sigma_points(
            prev_mean, prev_covariance
        )
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
        pred_mean, pred_covariance = self._unscented_transform.compute_distribution(
            pred_sigma_points
        )
        assert pred_mean.shape == (N, state_dim)
        assert pred_covariance.shape == (N, state_dim, state_dim)

        # Compute weighted covariances (see helper docstring for explanation)
        dynamics_covariance = self._weighted_covariance(
            pred_sigma_tril.reshape((N, sigma_point_count, state_dim, state_dim))
        )
        assert dynamics_covariance.shape == (N, state_dim, state_dim)
        measurement_covariance = self._weighted_covariance(
            pred_sigma_observations_tril.reshape(
                (N, sigma_point_count, observation_dim, observation_dim)
            )
        )
        assert measurement_covariance.shape == (N, observation_dim, observation_dim)

        # Add dynamics uncertainty
        pred_covariance = pred_covariance + dynamics_covariance

        return (
            pred_mean,
            pred_covariance,
            pred_sigma_points,
            pred_sigma_observations,
            measurement_covariance,
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
        # Extract inputs
        assert isinstance(
            observations, types.ObservationsNoDictTorch
        ), "For UKF, observations must be tensor!"
        observations = cast(types.ObservationsNoDictTorch, observations)
        (
            pred_mean,
            pred_covariance,
            pred_sigma_points,
            pred_sigma_observations,
            measurement_covariance,
        ) = predict_outputs

        # Check shapes
        N, sigma_point_count, state_dim = pred_sigma_points
        observation_dim = self.measurement_model.observation_dim
        assert pred_mean.shape == (N, state_dim)
        assert pred_covariance.shape == (N, state_dim, state_dim)
        assert pred_sigma_observations.shape == (N, sigma_point_count, observation_dim,)

        # Compute observation distribution
        (
            pred_observations,
            pred_observations_covariance,
        ) = self._unscented_transform.compute_distribution(pred_sigma_observations)
        assert pred_observations.shape == (N, observation_dim)
        assert pred_observations_covariance.shape == (
            N,
            observation_dim,
            observation_dim,
        )

        # Add measurement model uncertainty
        pred_observations_covariance = (
            pred_observations_covariance + measurement_covariance
        )

        # Compute cross-covariance
        centered_sigma_points = pred_sigma_points - pred_mean[:, None, :]
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
        K = cross_covariance @ torch.inverse(pred_observations_covariance)
        assert K.shape == (N, state_dim, observation_dim)

        innovations = observations - pred_observations

        # Update internal state with corrected beliefs
        self.belief_mean = pred_mean + K @ innovations
        self.belief_covariance = (
            pred_covariance - K @ pred_observations_covariance @ K.transpose(-1, -2)
        )

    def _weighted_covariance(self, sigma_trils: torch.Tensor):
        """For heteroscedastic covariances, we apply the weighted average approach
        described by Kloss et al:
        https://homes.cs.washington.edu/~barun/files/workshops/rss2020_sarl/submissions/7_differentiablefilter.pdf

        (note that the mean weights are used because they sum to 1)
        """
        N, sigma_point_count, dim, _ = sigma_trils.shape

        if sigma_trils.stride()[:2] == (0, 0):
            # All covariances identical => we can do less math
            output_covariance = sigma_trils[0] @ sigma_trils[0].transpose(-1, -2)
            assert output_covariance.shape == (dim, dim)
            output_covariance = output_covariance[None, :, :].expand((N, dim, dim))
        else:
            pred_sigma_tril = sigma_trils.reshape((N, sigma_point_count, dim, dim))
            pred_sigma_covariance = pred_sigma_tril @ pred_sigma_tril.transpose(-1, -2)
            output_covariance = torch.sum(
                self._unscented_transform.weights_m[None, :, None, None]
                * pred_sigma_covariance,
                dim=1,
            )
        return output_covariance
