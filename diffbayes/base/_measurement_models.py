import abc
from typing import Tuple

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


class VirtualSensorModel(abc.ABC, nn.Module):
    """Virtual sensor base class for our differentiable Kalman filters.

    Maps each observation input to a predicted state and covariance, in the style of
    BackpropKF. This is often necessary for complex observation spaces like images or
    point clouds, where it's not possible to learn a standard state->observation
    measurement model.

    To simplify logic, this class assumes that the measurement model is an identity.
    """

    def __init__(self, state_dim: int):
        super().__init__()

        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> Tuple[types.StatesTorch, types.CovarianceTorch]:
        """Predicts states from observation inputs.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing a state estimate and a
            covariance. Shapes should be `(N, state_dim)` and
            `(N, state_dim, state_dim)` respectively.
        """
        pass


class KalmanFilterMeasurementModel(abc.ABC, nn.Module):
    def __init__(self, *, state_dim, observation_dim):
        super().__init__()
        self.state_dim = state_dim
        """int: State dimensionality."""
        self.observation_dim = observation_dim
        """int: Observation dimensionality."""

    @abc.abstractmethod
    def forward(
        self, *, states: types.StatesTorch
    ) -> Tuple[types.ObservationsNoDictTorch, types.CovarianceTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected measurements
            and cholesky decomposition of covariance.  Shape should be `(N, M)`.
        """
        pass

    def jacobian(self, *, states: types.StatesTorch) -> torch.Tensor:
        """Returns Jacobian of the measurement model.

        Args:
            states (torch.Tensor): Current state, size `(N, state_dim)`.

        Returns:
            torch.Tensor: Jacobian, size `(N, observation_dim, state_dim)`
        """
        with torch.enable_grad():
            x = states.detach().clone()

            N, ndim = x.shape
            assert ndim == self.state_dim
            x = x.unsqueeze(1)
            x = x.repeat(1, ndim, 1)
            x.requires_grad_(True)
            y = self(states=x)

            mask = torch.eye(ndim).repeat(N, 1, 1).to(x.device)
            y = y[0]  # measurement model returns measurement first
            jac = torch.autograd.grad(y, x, mask, create_graph=True)

        return jac[0]
