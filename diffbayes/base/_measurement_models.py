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
        and just one unique observation.

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
    We assume that C matrix is identity.
    """

    def __init__(self, state_dim: int, R: torch.Tensor= None):
        super().__init__()

        self.state_dim = state_dim

        if R is not None:
            self.R = torch.nn.Parameter(R, requires_grad=False)

        """int: Dimensionality of our state."""

    @abc.abstractmethod
    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> (types.StatesTorch, types.CovarianceTorch):
        """Observation model forward pass, over batch size `N` to give us state estimation.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            states (torch.Tensor): States estimation from the observation.
            Shape should be `(N, state_dim)`.
            state_covariance (torch.Tensor): States covraiance estimation (R) from the observation.
            Shape should be `(N, state_dim, state_dim)`.
        """
        #todo: in crossmodal we also put in state input, changing it here.

        pass

    @abc.abstractmethod
    def get_C_matrix(self):
        #TODO: make into get function?

        return torch.nn.Parameter(torch.eye(self.state_dim), requires_grad=False)


