from typing import List

import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import fannypack

from .. import types


class ParticleFilterMeasurementDataset(torch.utils.data.Dataset):
    """A data preprocessor for pre-training particle filter measurement models.
    """

    def __init__(
        self,
        trajectories: List[types.TrajectoryTupleNumpy],
        *,
        covariance: np.ndarray,
        samples_per_pair,
        **kwargs
    ):
        """TODO
        """

        self.covariance = covariance
        self.samples_per_pair = samples_per_pair
        self.dataset = []

        # TODO: we can probably get rid of this for loop and just access trajectories
        # directly in __getitem__
        for i, trajectory in enumerate(tqdm(trajectories)):
            assert len(trajectory) == 3
            states, observations, controls = trajectory

            timesteps = len(states)
            assert type(observations) == dict
            assert len(controls) == timesteps

            for t in range(0, timesteps):
                # Pull out data & labels
                state = states[t]
                observation = fannypack.utils.SliceWrapper(observations)[t]
                self.dataset.append((state, observation))

        self.controls = controls
        self.observations = observations

        print("Loaded {} points".format(len(self.dataset)))

    def __getitem__(self, index):
        """TODO
        """

        state, observation = self.dataset[index // self.samples_per_pair]

        # Generate half of our samples close to the mean, and the other half
        # far away
        if index % self.samples_per_pair < self.samples_per_pair * 0.5:
            noisy_state = np.random.multivariate_normal(mean=state, cov=self.covariance)
        else:
            noisy_state = np.random.multivariate_normal(
                mean=state, cov=self.covariance * 5
            )

        log_likelihood = np.asarray(
            scipy.stats.multivariate_normal.logpdf(
                noisy_state, mean=state, cov=self.covariance
            )
        )

        return fannypack.utils.to_torch((noisy_state, observation, log_likelihood))

    def __len__(self):
        """Total number of samples in the dataset
        """
        return len(self.dataset) * self.samples_per_pair
