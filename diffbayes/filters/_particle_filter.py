import fannypack
import numpy as np
import torch
from overrides import overrides

from .. import types
from ..base._dynamics_model import DynamicsModel
from ..base._filter import Filter
from ..base._particle_filter_measurement_model import ParticleFilterMeasurementModel


class ParticleFilter(Filter):
    """Generic differentiable particle filter.
    """

    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        measurement_model: ParticleFilterMeasurementModel,
        num_particles: int = 100,
        resample: bool = None,
        soft_resample_alpha: float = 1.0,
        estimation_method: str = "weighted_average",
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)
        assert isinstance(measurement_model, ParticleFilterMeasurementModel)
        assert dynamics_model.state_dim == measurement_model.state_dim

        # Initialize state dimension
        state_dim = dynamics_model.state_dim
        super().__init__(state_dim=state_dim)

        # Assign submodules
        self.dynamics_model = dynamics_model
        """diffbayes.base.DynamicsModel: Forward model."""
        self.measurement_model = measurement_model
        """diffbayes.base.ParticleFilterMeasurementModel: Observation model."""

        # Settings
        self.num_particles = num_particles
        """int: Number of particles to represent our belief distribution.
        Defaults to 100."""
        self.resample = resample
        """bool: If True, we resample particles & normalize weights at each
        timestep. If unset (None), we automatically turn resampling on in eval mode
        and off in train mode."""

        self.soft_resample_alpha = soft_resample_alpha
        """float: Tunable constant for differentiable resampling, as described
        by Jonschkowski et al. in "Differentiable Particle Filters: End-to-End
        Learning with Algorithmic Priors": https://arxiv.org/abs/1805.11122.
        Defaults to 1.0 (disabled)."""

        assert estimation_method in ("weighted_average", "argmax")
        self.estimation_method = estimation_method
        """str: Method of producing state estimates. Options include:
        - 'weighted_average': average of particles weighted by their weights.
        - 'argmax': state of highest weighted particle.
        """

        # "Hidden state" tensors
        self.particle_states: torch.Tensor
        """torch.Tensor: Discrete particles representing our current belief
        distribution. Shape should be `(N, M, state_dim)`.
        """
        self.particle_log_weights: torch.Tensor
        """torch.Tensor: Weights corresponding to each particle, stored as
        log-likelihoods. Shape should be `(N, M)`.
        """
        self._initialized = False

    @overrides
    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: types.CovarianceTorch
    ):
        """Populates initial particles, which will be normally distributed.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        M = self.num_particles

        # Sample particles
        self.particle_states = (
            torch.distributions.MultivariateNormal(mean, covariance)
            .sample((M,))
            .transpose(0, 1)
        )
        assert self.particle_states.shape == (N, M, self.state_dim)

        # Normalize weights
        self.particle_log_weights = self.particle_states.new_full(
            (N, M), float(-np.log(M, dtype=np.float32))
        )
        assert self.particle_log_weights.shape == (N, M)

        # Set initialized flag
        self._initialized = True

    @overrides
    def forward(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Particle filter forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.

        Returns:
            torch.Tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Make sure our particle filter's been initialized
        assert self._initialized, "Particle filter not initialized!"

        # Get our batch size (N), current particle count (M), & state dimension
        N, M, state_dim = self.particle_states.shape
        assert state_dim == self.state_dim
        assert len(fannypack.utils.SliceWrapper(controls)) == N

        # Decide whether or not we're resampling
        resample = self.resample
        if resample is None:
            # If not explicitly set, we disable resampling in train mode (to allow
            # gradients to propagate through time) and enable in eval mode (to prevent
            # particle deprivation)
            resample = not self.training

        # If we're not resampling and our current particle count doesn't match
        # our desired particle count, we need to either expand or contract our
        # particle set
        if not resample and self.num_particles != M:
            indices = self.particle_states.new_zeros(
                (N, self.num_particles), dtype=torch.long
            )

            # If output particles > our input particles, for the beginning part we copy
            # particles directly to reduce variance
            copy_count = (self.num_particles // M) * M
            indices[:, :copy_count] = torch.arange(M).repeat(copy_count // M)[None, :]

            # For remaining particles, we sample w/o replacement (also lowers variance)
            remaining_count = self.num_particles - copy_count
            assert remaining_count >= 0
            if remaining_count > 0:
                uniform_logits = torch.ones_like(self.particle_log_weights[0])
                indices[:, copy_count:] = torch.stack(
                    [
                        torch.multinomial(
                            uniform_logits,
                            num_samples=remaining_count,
                            replacement=False,
                        )
                        for _ in range(N)
                    ],
                    dim=0,
                )

            # Gather new particles, weights
            new_states = self.particle_states.gather(
                1, indices[:, :, None].expand((N, self.num_particles, state_dim))
            )
            new_log_weights = self.particle_log_weights.gather(1, indices)
            assert new_states.shape == (N, self.num_particles, state_dim)
            assert new_log_weights.shape == (N, self.num_particles)

            # Update particle states and (normalized) weights
            self.particle_states = new_states
            self.particle_log_weights = new_log_weights - torch.logsumexp(
                new_log_weights, dim=1, keepdim=True
            )
            M = self.num_particles

        # Propagate particles through our dynamics model
        # A bit of extra effort is required for the extra particle dimension
        # > For our states, we flatten along the N/M axes
        # > For our controls, we repeat each one `M` times, if M=3:
        #       [u0 u1 u2] should become [u0 u0 u0 u1 u1 u1 u2 u2 u2]
        #
        # Currently each of the M particles within a "sample" get the same action, but
        # we could also add noise in the action space (a la Jonschkowski et al. 2018)
        reshaped_states = self.particle_states.reshape(-1, self.state_dim)
        reshaped_controls = fannypack.utils.SliceWrapper(controls).map(
            lambda tensor: torch.repeat_interleave(tensor, repeats=M, dim=0)
        )
        predicted_states, scale_trils = self.dynamics_model(
            initial_states=reshaped_states, controls=reshaped_controls
        )
        self.particle_states = (
            torch.distributions.MultivariateNormal(
                loc=predicted_states, scale_tril=scale_trils
            )
            .rsample()  # Note that we use `rsample` to make sampling differentiable
            .view(N, M, self.state_dim)
        )
        assert self.particle_states.shape == (N, M, self.state_dim)

        # Re-weight particles using observations
        self.particle_log_weights = self.particle_log_weights + self.measurement_model(
            states=self.particle_states, observations=observations,
        )

        # Normalize particle weights
        self.particle_log_weights = self.particle_log_weights - torch.logsumexp(
            self.particle_log_weights, dim=1, keepdim=True
        )

        # Compute output
        if self.estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(self.particle_log_weights[:, :, np.newaxis])
                * self.particle_states,
                dim=1,
            )
        elif self.estimation_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=1)
            state_estimates = torch.gather(
                self.particle_states, dim=1, index=best_indices
            )
        else:
            assert False, "Unsupported estimation method!"

        # Resampling
        if resample:
            if self.soft_resample_alpha < 1.0:
                # TODO: port this implementation over!
                assert False, "Not yet ported"
            else:
                # Standard particle filter re-sampling -- this stops gradients
                # This is the most naive flavor of resampling, and not the low
                # variance approach
                #
                # Note the distinction between M, the current # of particles,
                # and self.num_particles, the desired # of particles
                assert self.particle_log_weights.shape == (N, M)
                distribution = torch.distributions.Categorical(
                    logits=self.particle_log_weights
                )
                state_indices = distribution.sample((self.num_particles,)).T
                assert state_indices.shape == (N, self.num_particles)

                new_states = torch.zeros_like(self.particle_states)
                for i in range(N):
                    # TODO: 90% sure this loop can be factored out
                    new_states[i] = self.particle_states[i][state_indices[i]]

                # Uniform weights
                self.particle_log_weights = self.particle_log_weights.new_full(
                    (N, self.num_particles), float(-np.log(M, dtype=np.float32))
                )

        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates
