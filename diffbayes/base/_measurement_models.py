import abc

import torch
import torch.nn as nn

from .. import types


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


class KalmanFilterMeasurementModel(abc.ABC, nn.Module):
    """Observation model base class for a generic differentiable Kalman filter;
    maps (observation) pairs to a state estimation.

    TODO: This is technically a virtual sensor instead of a measurement model.
    """

    def __init__(self, state_dim: int, noise_R_tril: torch.Tensor= None):
        super().__init__()

        self.state_dim = state_dim
        """int: Dimensionality of our state."""

        if noise_R_tril is not None:
            self.noise_R_tril = torch.nn.Parameter(noise_R,
                                              requires_grad=False)

    @abc.abstractmethod
    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> (types.StatesTorch, types.CovarianceTorch):
        """Observation model forward pass, over batch size `N` to give us state estimation.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            measurement (torch.Tensor): State estimation from the observation.
            Shape should be `(N, state_dim)`.
            covariance (torch.Tensor): Measurrement covariance (R) from the observation.
            Shape should be `(N, state_dim, state_dim)`.
        """

        pass
