import abc
import torch.nn as nn
import fannypack


class DynamicsModel(nn.Module, abc.ABC):
    """Base class for a generic differentiable dynamics model.

    As a minimum, subclasses should override either `forward` or `forward_loop`
    for computing dynamics estimates.
    """

    def __init__(self, *, state_dim):
        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    def forward(self, *, initial_states, controls):
        """Dynamics model forward pass, single timestep.

        By default, this is implemented by bootstrapping the `forward_loop()`
        method.

        Args:
            initial_states (torch.tensor): Initial states of our system.
            controls (dict or torch.tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            torch.tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Wrap our control inputs
        #
        # If the input is a dictionary of tensors, this provides a
        # tensor-like interface for slicing, accessing shape, etc
        controls = fannypack.utils.SliceWrapper(controls)

        # Call `forward_loop()` with a single timestep
        output = self.forward_loop(
            initial_states=initial_states, controls=controls[None, ...]
        )
        assert output.shape[0] == 1
        return output[0]

    def forward_loop(self, *, initial_states, controls):
        """Dynamics model forward pass, over sequence length `T` and batch size
        `N`.  By default, this is implemented by iteratively calling
        `forward()`.

        To inject code between timesteps (for example, to inspect hidden state),
        use `register_forward_hook()`.

        Args:
            initial_states (torch.tensor): Initial states to pass to our
                dynamics model. Shape should be `(N, state_dim)`.
            controls (dict or torch.tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(T, N, ...)`.

        Returns:
            torch.tensor: Predicted states at each timestep. Shape should be
            `(T, N, state_dim).`
        """

        # Wrap our control inputs
        #
        # If the input is a dictionary of tensors, this provides a
        # tensor-like interface for slicing, accessing shape, etc
        controls = fannypack.utils.SliceWrapper(controls)

        # Get sequence length (T), batch size (N)
        T = controls.shape[0]
        N = controls.shape[1]
        assert initial_states.shape == (N, self.state_dim)

        # Dynamics forward pass
        state_predictions = torch.zeros((T, N, self.state_dim))
        current_estimate = initial_states
        for t in range(T):
            # Compute state estimate for a single timestep
            # We use __call__ to make sure hooks are dispatched correctly
            current_estimate = self(
                initial_states=current_estimate, controls=controls[t],
            )

            # Validate & add to output
            assert current_estimate.shape == (N, self.state_dim)
            state_predictions[t] = current_estimate

        # Return state estimates
        return state_predictions
