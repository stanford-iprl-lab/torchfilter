import abc

import torch.nn as nn


class ParticleFilterMeasurementModel(abc.ABC, nn.Module):
    """Observation model base class for a generic differentiable particle
    filter; maps (state, observation) pairs to the log-likelihood of the
    observation given the state ( $\\log p(z | x)$ ).
    """

    def __init__(self, state_dim):
        self.state_dim = state_dim
        """int: Dimensionality of our state."""

    @abc.abstractmethod
    def forward(self, *, states, observations):
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect `M` separate states (particles)
        and just one unique observation.

        Args:
            states (torch.tensor): States to pass to our observation model.
                Shape should be `(N, M, state_dim)`.
            observations (dict or torch.tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            torch.tensor: Log-likelihoods of each state, conditioned on a
            corresponding observation. Shape should be `(N, M)`.
        """
        pass
