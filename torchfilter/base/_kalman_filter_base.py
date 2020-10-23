"""Private module; avoid importing from directly.
"""

import abc

import fannypack as fp
import torch
from overrides import overrides

from .. import types
from ._dynamics_model import DynamicsModel
from ._filter import Filter
from ._kalman_filter_measurement_model import KalmanFilterMeasurementModel


class KalmanFilterBase(Filter, abc.ABC):
    """Base class for a generic Kalman-style filter. Parameterizes beliefs with a mean
    and covariance.

    Subclasses should override `_predict_step()` and `_update_step()`.
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: KalmanFilterMeasurementModel,
        **unused_kwargs,  # For type checking
    ):
        super().__init__(state_dim=dynamics_model.state_dim)

        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, KalmanFilterMeasurementModel)

        # Assign submodules
        self.dynamics_model = dynamics_model
        """torchfilter.base.DynamicsModel: Forward model."""

        self.measurement_model = measurement_model
        """torchfilter.base.KalmanFilterMeasurementModel: Measurement model."""

        # Protected attributes for posterior distribution: these should be accessed
        # through the public `.belief_mean` and `.belief_covariance` properties
        #
        # `_belief_covariance` is unused for square-root filters
        self._belief_mean: torch.Tensor
        self._belief_covariance: torch.Tensor

        # Throw an error if our filter is used before `.initialize_beliefs()` is called
        self._initialized = False

    @overrides
    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
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
        assert self._initialized, "Kalman filter not initialized!"

        # Validate inputs
        N, state_dim = self.belief_mean.shape
        assert fp.utils.SliceWrapper(observations).shape[0] == N
        assert fp.utils.SliceWrapper(controls).shape[0] == N

        # Predict step
        self._predict_step(controls=controls)

        # Update step
        self._update_step(observations=observations)

        # Return mean
        return self.belief_mean

    @overrides
    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: types.CovarianceTorch
    ) -> None:
        """Set filter belief to a given mean and covariance.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        self.belief_mean = mean
        self.belief_covariance = covariance
        self._initialized = True

    @property
    def belief_mean(self) -> types.StatesTorch:
        """Posterior mean. Shape should be `(N, state_dim)`."""
        return self._belief_mean

    @belief_mean.setter
    def belief_mean(self, mean: types.StatesTorch):
        self._belief_mean = mean

    @property
    def belief_covariance(self) -> types.CovarianceTorch:
        """Posterior covariance. Shape should be `(N, state_dim, state_dim)`."""
        return self._belief_covariance

    @belief_covariance.setter
    def belief_covariance(self, covariance: types.CovarianceTorch):
        self._belief_covariance = covariance

    @abc.abstractmethod
    def _predict_step(self, *, controls: types.ControlsTorch) -> None:
        r"""Kalman filter predict step.

        Computes $\mu_{t | t - 1}$, $\Sigma_{t | t - 1}$ from $\mu_{t - 1 | t - 1}$,
        $\Sigma_{t - 1 | t - 1}$.

        Keyword Args:
            controls (dict or torch.Tensor): Control inputs.
        """

    @abc.abstractmethod
    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        r"""Kalman filter measurement update step.

        Nominally, computes $\mu_{t | t}$, $\Sigma_{t | t}$ from $\mu_{t | t - 1}$,
        $\Sigma_{t | t - 1}$.

        Updates `self.belief_mean` and `self.belief_covariance`.

        Keyword Args:
            observations (dict or torch.Tensor): Observation inputs.
        """
