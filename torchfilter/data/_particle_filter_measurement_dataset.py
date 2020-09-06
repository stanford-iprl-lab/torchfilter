"""Private module; avoid importing from directly.
"""

from typing import List, Tuple

import fannypack
import numpy as np
import scipy.stats
import torch
from tqdm.auto import tqdm

from .. import types


class ParticleFilterMeasurementDataset(torch.utils.data.Dataset):
    """A dataset interface for pre-training particle filter measurement models.

    Centers Gaussian distributions around our ground-truth states, and provides examples
    for learning the log-likelihood.

    Args:
        trajectories (List[torchfilter.types.TrajectoryNumpy]): List of trajectories.

    Keyword Args:
        covariance (np.ndarray): Covariance of Gaussian PDFs.
        samples_per_pair (int): Number of training examples to provide for each
            state/observation pair. Half of these will typically be generated close
            to the example, and the other half far away.
    """

    def __init__(
        self,
        trajectories: List[types.TrajectoryNumpy],
        *,
        covariance: np.ndarray,
        samples_per_pair: int,
        **kwargs
    ):
        self.covariance = covariance.astype(np.float32)
        self.samples_per_pair = samples_per_pair
        self.dataset = []
        self.rng = np.random.default_rng()

        for i, traj in enumerate(tqdm(trajectories)):
            T = len(traj.states)
            assert len(traj.controls) == T

            for t in range(T):
                # Pull out data & labels
                state = traj.states[t]
                observation = fannypack.utils.SliceWrapper(traj.observations)[t]
                self.dataset.append((state, observation))

        self.controls = traj.controls
        self.observations = traj.observations

        print("Loaded {} points".format(len(self.dataset)))

    def __getitem__(
        self, index
    ) -> Tuple[types.StatesNumpy, types.ObservationsNumpy, np.ndarray]:
        """Get a state/observation/log-likelihood sample from our dataset. Nominally, we
        want our measurement model to predict the returned log-likelihood as the PDF of
        the `p(observation | state)` distribution.

        Args:
            index (int): Subsequence number in our dataset.
        Returns:
            tuple: `(state, observation, log-likelihood)` tuple.
        """

        state, observation = self.dataset[index // self.samples_per_pair]

        # Generate half of our samples close to the mean, and the other half
        # far away
        if index % self.samples_per_pair < self.samples_per_pair * 0.5:
            noisy_state = self.rng.multivariate_normal(
                mean=state, cov=self.covariance
            ).astype(np.float32)
        else:
            noisy_state = self.rng.multivariate_normal(
                mean=state, cov=self.covariance * 5
            ).astype(np.float32)

        log_likelihood = np.asarray(
            scipy.stats.multivariate_normal.logpdf(
                noisy_state, mean=state, cov=self.covariance
            ),
            dtype=np.float32,
        )

        return noisy_state, observation, log_likelihood

    def __len__(self) -> int:
        """Total number of samples in the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.dataset) * self.samples_per_pair
