"""Private module; avoid importing from directly.
"""

import abc

import fannypack
import torch.nn as nn
from overrides import overrides

from .. import types


class Filter(nn.Module, abc.ABC):
    """Base class for a generic differentiable state estimator.

    As a minimum, subclasses should override:
    - `initialize_beliefs` for populating the initial belief of our estimator
    - `forward` or `forward_loop` for computing state predictions
    """

    def __init__(self, *, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: types.CovarianceTorch
    ) -> None:
        """Initialize our filter with a Gaussian prior.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        pass

    @overrides
    def forward(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch
    ) -> types.StatesTorch:
        """Filtering forward pass, over a single timestep.

        By default, this is implemented by bootstrapping the `forward_loop()`
        method.

        Args:
            observations (dict or torch.Tensor): Observation inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Wrap our observation and control inputs
        #
        # If either of our inputs are dictionaries, this provides a tensor-like
        # interface for slicing, accessing shape, etc
        observations_wrapped = fannypack.utils.SliceWrapper(observations)
        controls_wrapped = fannypack.utils.SliceWrapper(controls)

        # Call `forward_loop()` with a single timestep
        output = self.forward_loop(
            observations=observations_wrapped[None, ...],
            controls=controls_wrapped[None, ...],
        )
        assert output.shape[0] == 1
        return output[0]

    def forward_loop(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch
    ) -> types.StatesTorch:
        """Filtering forward pass, over sequence length `T` and batch size `N`.
        By default, this is implemented by iteratively calling `forward()`.

        To inject code between timesteps (for example, to inspect hidden state),
        use `register_forward_hook()`.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of size `(T, N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of size `(T, N, ...)`.

        Returns:
            torch.Tensor: Predicted states at each timestep. Shape should be
            `(T, N, state_dim).`
        """

        # Wrap our observation and control inputs
        #
        # If either of our inputs are dictionaries, this provides a tensor-like
        # interface for slicing, accessing shape, etc
        observations_wrapped = fannypack.utils.SliceWrapper(observations)
        controls_wrapped = fannypack.utils.SliceWrapper(controls)

        # Get sequence length (T), batch size (N)
        T, N = controls_wrapped.shape[:2]
        assert observations_wrapped.shape[:2] == (T, N)

        # Filtering forward pass
        # We treat t = 0 as a special case to make it easier to create state_predictions
        # tensor on the correct device
        t = 0
        current_prediction = self(
            observations=observations_wrapped[t], controls=controls_wrapped[t]
        )
        state_predictions = current_prediction.new_zeros((T, N, self.state_dim))
        assert current_prediction.shape == (N, self.state_dim)
        state_predictions[t] = current_prediction

        for t in range(1, T):
            # Compute state prediction for a single timestep
            # We use __call__ to make sure hooks are dispatched correctly
            current_prediction = self(
                observations=observations_wrapped[t], controls=controls_wrapped[t]
            )

            # Validate & add to output
            assert current_prediction.shape == (N, self.state_dim)
            state_predictions[t] = current_prediction

        # Return state predictions
        return state_predictions
