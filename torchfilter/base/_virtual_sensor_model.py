"""Private module; avoid importing from directly.
"""

import abc
from typing import Tuple

import torch.nn as nn
from overrides import overrides

from .. import types


class VirtualSensorModel(abc.ABC, nn.Module):
    """Virtual sensor base class for our differentiable Kalman filters.

    Maps each observation input to a predicted state and uncertainty, in the style of
    BackpropKF. This is often necessary for complex observation spaces like images or
    point clouds, where it's not possible to learn a standard state->observation
    measurement model.
    """

    def __init__(self, state_dim: int):
        super().__init__()

        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    @overrides
    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Predicts states and uncertainties from observation inputs.

        Uncertainties should be lower-triangular Cholesky decompositions of covariance
        matrices.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties. States
            should have shape `(N, state_dim)`, and uncertainties should be lower
            triangular with shape `(N, state_dim, state_dim).`
        """
