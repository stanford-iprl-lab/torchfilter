import abc
from typing import Any, Optional, Tuple

import torch

import fannypack as fp

from .. import types
from ._filter import Filter


class KalmanFilterBase(Filter, abc.ABC):
    """Base class for a generic Kalman-style filter. Parameterizes beliefs with a mean
    and covariance.

    Subclasses should override _predict_step() and _update_step().
    """

    def __init__(self, *, state_dim: int):
        super().__init__(state_dim=state_dim)

        self.belief_mu: Optional[torch.Tensor] = None
        """torch.Tensor: Estimated state mean."""
        self.belief_cov: Optional[torch.Tensor] = None
        """torch.Tensor: Estimated state covariance."""

    def forward(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Kalman filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): Observation inputs. Should be either a
            dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): Control inputs. Should be either a dict of
            tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """
        # Check initialization
        assert (
            self.belief_mu is not None and self.belief_cov is not None
        ), "Kalman filter not initialized!"

        # Validate inputs
        N, state_dim = self.belief_mu.shape
        assert fp.utils.SliceWrapper(observations).shape[0] == N
        assert fp.utils.SliceWrapper(controls).shape[0] == N

        # Predict step
        # It's unfortunately not possible to make these helpers more functional, because
        # the requirements of each filter are pretty different. (particularly for UKFs,
        # square root formulations, etc)
        predict_outputs = self._predict_step(controls=controls)

        # Update step
        self._update_step(predict_outputs=predict_outputs, observations=observations)

        # Validate belief
        assert self.belief_mu.shape == (N, state_dim)
        assert self.belief_cov.shape == (N, state_dim, state_dim)

        return self.belief_mu

    def initialize_beliefs(self, *, mean: torch.Tensor, covariance: torch.Tensor):
        """Set filter belief to a given mean and covariance.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        self.belief_mu = mean
        self.belief_cov = covariance

    @abc.abstractmethod
    def _predict_step(self, *, controls: types.ControlsTorch) -> Any:
        r"""Kalman filter predict step.

        Computes $\mu_{t | t - 1}$, $\Sigma_{t | t - 1}$ from $\mu_{t - 1 | t - 1}$,
        $\Sigma_{t - 1 | t - 1}$.

        Keyword Args:
            controls (dict or torch.Tensor): Control inputs.

        Returns:
            Any: Predict outputs, to pass to update step.
        """
        pass

    @abc.abstractmethod
    def _update_step(
        self, *, predict_outputs: Any, observations: types.ObservationsTorch
    ) -> None:
        r"""Kalman filter measurement update step.

        Nominally, computes $\mu_{t | t}$, $\Sigma_{t | t}$ from $\mu_{t | t - 1}$,
        $\Sigma_{t | t - 1}$.

        Updates `self.belief_mu` and `self.belief_cov`.

        Keyword Args:
            predict_outputs (Any): Outputs from predict step.
            observations (dict or torch.Tensor): Observation inputs.
        """
        pass