from typing import List, Tuple

import fannypack as fp
from torch.utils.data import Dataset

from .. import types


class SingleStepDataset(Dataset):
    """A dataset interface that returns single-step training examples:
    `(previous_state, state, observation, control)`

    By default, extracts these examples from a list of trajectories.

    Args:
        trajectories (List[diffbayes.types.TrajectoryNumpy]): List of trajectories.
    """

    def __init__(self, trajectories: List[types.TrajectoryNumpy]):
        self.samples: List[
            Tuple[
                types.StatesNumpy,
                types.StatesNumpy,
                types.ObservationsNumpy,
                types.ControlsNumpy,
            ]
        ] = []

        for traj in trajectories:
            T = len(traj.states)
            for t in range(T - 1):
                self.samples.append(
                    (
                        traj.states[t],  # previous_state
                        traj.states[t + 1],  # state
                        fp.utils.SliceWrapper(traj.observations)[t + 1],  # observation
                        fp.utils.SliceWrapper(traj.controls)[t + 1],  # control
                    )
                )

    def __getitem__(
        self, index: int
    ) -> Tuple[
        types.StatesNumpy,
        types.StatesNumpy,
        types.ObservationsNumpy,
        types.ControlsNumpy,
    ]:
        """Get a single-step prediction sample from our dataset.

        Args:
            index (int): Subsequence number in our dataset.
        Returns:
            tuple: `(previous_state, state, observation, control)` tuple that
            contains data for a single subsequence. Each tuple member should be either a
            numpy array or dict of numpy arrays with shape `(subsequence_length, ...)`.
        """
        return self.samples[index]

    def __len__(self) -> int:
        """Total number of subsequences in the dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.samples)
