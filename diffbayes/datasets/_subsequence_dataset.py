from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import types as types
from ._split_trajectories import split_trajectories


class SubsequenceDataset(Dataset):
    """A data preprocessor for producing training subsequences from
    a list of trajectories.

    Thin wrapper around `diffbayes.datasets.split_trajectories()`.

    Args:
        trajectories (list): list of trajectories, where each is a tuple of
            `(states, observations, controls)`. Each tuple member should be
            either a numpy array or dict of numpy arrays with shape `(T, ...)`.
        subsequence_length (int): # of timesteps per subsequence.
    """

    def __init__(
        self, trajectories: List[types.TrajectoryTupleNumpy], subsequence_length: int
    ):
        # Split trajectory into overlapping subsequences
        self.subsequences: List[types.TrajectoryTupleNumpy] = split_trajectories(
            trajectories, subsequence_length
        )

    def __getitem__(self, index: int) -> types.TrajectoryTupleNumpy:
        """Get a subsequence from our dataset.

        Args:
            index (int): Subsequence number in our dataset.
        Returns:
            tuple: `(states, observations, controls)` tuple that contains
            data for a single subsequence. Each tuple member should be either a
            numpy array or dict of numpy arrays with shape
            `(subsequence_length, ...)`.
        """
        states, observation, controls = self.subsequences[index]
        return states, observation, controls

    def __len__(self):
        """Total number of subsequences in the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.subsequences)
