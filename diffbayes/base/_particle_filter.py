import abc

import numpy as np
import torch

from . import DynamicsModel, Filter, ParticleFilterMeasurementModel


class ParticleFilter(Filter):
    """Base class for a generic differentiable particle filter.
    """

    def __init__(
        self,
        *,
        dynamics_model,
        measurement_model,
        num_particles=100,
        resample=True,
        soft_resample_alpha=1.0,
        estimation_method="weighted_average",
    ):
        # Sanity check for submodules
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
        """int: If True, we resample particles & normalize weights at each
        timestep."""

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
        self.particle_states = None
        """torch.tensor: Discrete particles representing our current belief
        distribution. Shape should be `(N, M, state_dim)`.
        """
        self.particle_log_weights = None
        """torch.tensor: Weights corresponding to each particle, stored as
        log-likelihoods. Shape should be `(N, M)`.
        """

    def initialize_beliefs(self, *, mean, covariance):
        """Populates initial particles, which will be normally distributed.

        Args:
            mean (torch.tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)
        M = self.num_particles

        # Sample particles
        self.particle_states = (
            torch.distributions.MultivariateNormal(mean, covariance)
            .sample(M)
            .transpose(0, 1)
        )
        assert particle_states.shape == (N, M, self.state_dim)

        # Normalize weights
        self.particle_log_weights = self.particle_states.new_full(
            (N, M), np.log(M, dtype=np.float32)
        )
        assert self.particle_log_weights == (N, M)

    def forward(self, *, observations, controls):
        """Particle filter forward pass, single timestep.

        Args:
            observations (dict or torch.tensor): observation inputs. should be
                either a dict of tensors or tensor of size `(N, ...)`.
            controls (dict or torch.tensor): control inputs. should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            torch.tensor: Predicted state for each batch element. Shape should
            be `(N, state_dim).`
        """

        # Make sure our particle filter's been initialized
        assert (
            self.particle_states != None and self.particle_log_weights != None
        ), "Particle filter not initialized!"

        # Get our batch size (N), current particle count (M), & state dimension
        N, M, state_dim = self.particles.shape
        assert state_dim == self.state_dim

        # If we're not resampling and our current particle count doesn't match
        # our desired particle count, we need to either expand or contract our
        # particle set
        #
        # This is rarely needed and done fairly naively
        if not self.resample and self.num_particles != M:
            new_states = self.particle_states.new_zeros(
                (N, self.num_particles, state_dim)
            )
            new_log_weights = self.particle_log_weights.new_zeros(
                (N, self.num_particles)
            )

            # Randomly sample some particles from our input
            # We sample with replacement only if necessary
            uniform_logits = torch.ones_like(self.particle_log_weights[0])
            for i in range(N):
                # TODO: 90% sure this for loop can be factored out
                indices = torch.multinomial(
                    uniform_logits,
                    num_samples=self.num_particles,
                    replacement=(self.num_particles > M),
                )
                new_states[i] = self.particle_states[i][indices]
                new_log_weights[i] = self.particle_log_weights[i][indices]

            # Update particle states and (normalized) weights
            self.particle_states = new_states
            self.particle_log_weights = new_log_weights - torch.logsumexp(
                new_log_weights, dim=1, keepdim=True
            )
            M = self.num_particles

        # Propagate particles through our dynamics model
        # A bit of extra effort is required for the extra particle dimension
        self.particle_states = self.dynamics_model(
            initial_states=self.particle_states.view(-1, self.state_dim),
            controls=controls,
        ).view(N, M, self.state_dim)
        assert self.particle_states == (N, M, self.state_dim)

        # Re-weight particles using observations
        self.particle_log_weights = self.particle_log_weights + self.measurement_model(
            states=self.particle_states, observations=observations
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
        elif self.estimattion_method == "argmax":
            best_indices = torch.argmax(self.particle_log_weights, dim=1)
            state_estimates = torch.gather(
                self.particle_states, dim=1, index=best_indices
            )
        else:
            assert False, "Invalid estimation method!"

        # Resampling
        if resample:
            if self.soft_resample_alpha < 1.0:
                # TODO: port this implementation over!
                assert False, "Not yet ported"
            else:
                # Standard particle filter re-sampling -- this kills gradients
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

                new_states = torch.zeros_like(states_pred)
                for i in range(N):
                    # TODO: 90% sure this loop can be factored out
                    new_states[i] = self.particle_states[i][state_indices[i]]

                # Uniform weights
                self.particle_log_weights = self.particle_log_weights.new_full(
                    (N, self.num_particles), np.log(M, dtype=np.float32)
                )

        # Post-condition :)
        assert state_estimates.shape == (N, state_dim)
        assert self.particle_states.shape == (N, self.num_particles, state_dim)
        assert self.particle_log_weights.shape == (N, self.num_particles)

        return state_estimates
