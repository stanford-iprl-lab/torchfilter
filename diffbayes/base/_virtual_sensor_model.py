import abc
from typing import Tuple

import torch.nn as nn

from .. import types


class VirtualSensorModel(abc.ABC, nn.Module):
    """Virtual sensor base class for our differentiable Kalman filters.

    Maps each observation input to a predicted state and covariance, in the style of
    BackpropKF. This is often necessary for complex observation spaces like images or
    point clouds, where it's not possible to learn a standard state->observation
    measurement model.
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
