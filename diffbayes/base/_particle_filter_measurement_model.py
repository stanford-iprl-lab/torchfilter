import abc
from typing import cast

import torch
import torch.nn as nn
from overrides import overrides

from .. import types
from ._kalman_filter_measurement_model import KalmanFilterMeasurementModel


class ParticleFilterMeasurementModel(abc.ABC, nn.Module):
    """Observation model base class for a generic differentiable particle
    filter; maps (state, observation) pairs to the log-likelihood of the
    observation given the state ( $\\log p(z | x)$ ).
    """

    def __init__(self, state_dim: int):
        super().__init__()

        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    @overrides
    def forward(
        self, *, states: types.StatesTorch, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect `M` separate states (particles)
        and one unique observation.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, M, state_dim)`.
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            torch.Tensor: Log-likelihoods of each state, conditioned on a
            corresponding observation. Shape should be `(N, M)`.
        """
        pass


class ParticleFilterMeasurementModelWrapper(ParticleFilterMeasurementModel):
    """Helper class for creating a particle filter measurement model (states,
    observations -> log-likelihoods) from a Kalman filter one (states -> observations).

    Args:
        kalman_filter_measurement_model (KalmanFilterMeasurementModel): Kalman filter
            measurement model instance to wrap.
    """

    def __init__(self, kalman_filter_measurement_model: KalmanFilterMeasurementModel):
        super().__init__(state_dim=kalman_filter_measurement_model.state_dim)
        self.kalman_filter_measurement_model = kalman_filter_measurement_model

    @overrides
    def forward(
        self, *, states: types.StatesTorch, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect `M` separate states (particles)
        and one unique observation.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, M, state_dim)`.
            observations (torch.Tensor): Measurement inputs. Should be either a dict of
                tensors or tensor of size `(N, ...)`.
        Returns:
            torch.Tensor: Log-likelihoods of each state, conditioned on a
            corresponding observation. Shape should be `(N, M)`.
        """

        # Note that Kalman filter measurement models only accept tensors as observation
        # inputs.
        assert isinstance(
            observations, torch.Tensor
        ), "For wrapped Kalman filter measurement models, observations must be tensors."
        observations = cast(types.ObservationsNoDictTorch, observations)

        # Shape checks
        N, M, state_dim = states.shape
        N_alt, observation_dim = observations.shape
        assert observation_dim == self.kalman_filter_measurement_model.observation_dim
        assert N == N_alt

        # Get predicted observations
        pred_observations, observations_tril = self.kalman_filter_measurement_model(
            states=states.reshape((-1, state_dim))
        )
        assert pred_observations.shape == (N * M, observation_dim)
        assert observations_tril.shape == (N * M, observation_dim, observation_dim)

        # Expand observations to account for particle count
        # This is currently not very memory-efficient
        observations = torch.repeat_interleave(observations, repeats=M, dim=0)
        assert observations.shape == (N * M, observation_dim)

        # Compute log likelihoods
        log_likelihoods = torch.distributions.MultivariateNormal(
            loc=pred_observations, scale_tril=observations_tril
        ).log_prob(observations)
        assert log_likelihoods.shape == (N * M,)

        # Reshape and return
        return log_likelihoods.reshape(N, M)
