from typing import List, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class SingleStepDataset(Dataset):
    """A data preprocessor for producing single-step training examples from
    a list of trajectories.

    Args:
        trajectories (list): list of trajectories, where each is a tuple of
            `(states, observations, controls)`. Each tuple member should be
            either a numpy array or dict of numpy arrays with shape `(T, ...)`.
    """

    def __init__(self, trajectories: List[Tuple]):
        # Split trajectory into samples:
        #   (initial_state, next_state, observation, control)
        self.samples = []

        for trajectory in trajectories:
            states, observations, controls = trajectory
            timesteps = len(states)
            for t in range(timesteps - 1):
                self.samples.append(
                    (
                        states[t],  # initial_state
                        states[t + 1],  # next_state
                        observations[t + 1],  # observation
                        controls[t + 1],  # control
                    )
                )

    def __getitem__(
        self, index: int
    ) -> Tuple[
        np.ndarray, np.ndarray, Union[dict, np.ndarray], Union[dict, np.ndarray]
    ]:
        """Get a single-step prediction sample from our dataset.

        Args:
            index (int): Subsequence number in our dataset.
        Returns:
            tuple: `(initial_state, next_state, observation, control)` tuple that
            contains data for a single subsequence. Each tuple member should be either a
            numpy array or dict of numpy arrays with shape `(subsequence_length, ...)`.
        """
        return self.samples[index]

    def __len__(self):
        """Total number of subsequences in the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.samples)
