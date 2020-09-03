import abc
from typing import Tuple

import torch
import torch.nn as nn

from .. import types


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
    ) -> Tuple[types.ObservationsNoDictTorch, types.ScaleTrilTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected observations
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
        observation_dim = self.observation_dim
        with torch.enable_grad():
            x = states.detach().clone()

            N, ndim = x.shape
            assert ndim == self.state_dim
            x = x[:, None, :].expand((N, observation_dim, ndim))
            x.requires_grad_(True)
            y = self(states=x.reshape((-1, ndim)))[0].reshape((N, -1, observation_dim))
            mask = torch.eye(observation_dim, device=x.device).repeat(N, 1, 1)
            jac = torch.autograd.grad(y, x, mask, create_graph=True)

        return jac[0]
