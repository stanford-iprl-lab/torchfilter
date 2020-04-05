import abc
import torch
import torch.nn as nn
import fannypack


class Filter(nn.Module, abc.ABC):
    """Base class for a generic differentiable state estimator.

    As a minimum, subclasses should override:
    - `initialize_beliefs` for populating the initial belief of our estimator
    - `forward` or `forward_loop` for computing state estimates
    """

    def __init__(self, *, state_dim):
        super().__init__()
        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    def initialize_beliefs(self, *, mean, covariance):
        """Initialize our filter with a Gaussian prior.

        Args:
            mean (torch.tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        pass

    def forward(self, *, observations, controls):
        """Filtering forward pass, over a single timestep.

        By default, this is implemented by bootstrapping the `forward_loop()`
        method.

        Args:
            observations (dict or torch.tensor): Observation inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
            controls (dict or torch.tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            torch.tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Wrap our observation and control inputs
        #
        # If either of our inputs are dictionaries, this provides a tensor-like
        # interface for slicing, accessing shape, etc
        observations = fannypack.utils.SliceWrapper(observations)
        controls = fannypack.utils.SliceWrapper(controls)

        # Call `forward_loop()` with a single timestep
        output = self.forward_loop(
            observations=observations[None, ...], controls=controls[None, ...]
        )
        assert output.shape[0] == 1
        return output[0]

    def forward_loop(self, *, observations, controls):
        """Filtering forward pass, over sequence length `T` and batch size `N`.
        By default, this is implemented by iteratively calling `forward()`.

        To inject code between timesteps (for example, to inspect hidden state),
        use `register_forward_hook()`.

        Args:
            observations (dict or torch.tensor): observation inputs. should be
                either a dict of tensors or tensor of size `(T, N, ...)`.
            controls (dict or torch.tensor): control inputs. should be either a
                dict of tensors or tensor of size `(T, N, ...)`.

        Returns:
            torch.tensor: Predicted states at each timestep. Shape should be
            `(T, N, state_dim).`
        """

        # Wrap our observation and control inputs
        #
        # If either of our inputs are dictionaries, this provides a tensor-like
        # interface for slicing, accessing shape, etc
        observations = fannypack.utils.SliceWrapper(observations)
        controls = fannypack.utils.SliceWrapper(controls)

        # Get sequence length (T), batch size (N)
        T = controls.shape[0]
        N = controls.shape[1]
        assert observations.shape[:2] == (T, N)

        # Filtering forward pass
        state_predictions = torch.zeros((T, N, self.state_dim))
        for t in range(T):
            # Compute state estimate for a single timestep
            # We use __call__ to make sure hooks are dispatched correctly
            current_estimate = self(observations[t], controls[t])

            # Validate & add to output
            assert current_estimate.shape == (N, state_dim)
            state_predictions[t] = current_estimate

        # Return state estimates
        return state_predictions
