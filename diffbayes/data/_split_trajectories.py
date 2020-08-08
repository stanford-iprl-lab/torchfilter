from typing import Dict, List, cast

import numpy as np

import fannypack as fp

from .. import types


def split_trajectories(
    trajectories: List[types.TrajectoryNumpy], subsequence_length: int
) -> List[types.TrajectoryNumpy]:
    """Helper for splitting a list of trajectories into a list of overlapping
    subsequences.

    For each trajectory, assuming a subsequence length of 10, this function
    includes in its output overlapping subsequences corresponding to
    timesteps...
    ```
        [0:10], [10:20], [20:30], ...
    ```
    as well as...
    ```
        [5:15], [15:25], [25:30], ...
    ```

    Args:
        trajectories (List[diffbayes.base.TrajectoryNumpy]): List of trajectories.
        subsequence_length (int): # of timesteps per subsequence.
    Returns:
        List[diffbayes.base.TrajectoryNumpy]: List of subsequences.
    """

    subsequences = []

    for traj in trajectories:
        # Chop up each trajectory into overlapping subsequences
        trajectory_length = len(traj.states)
        assert len(fp.utils.SliceWrapper(traj.observations)) == trajectory_length
        assert len(fp.utils.SliceWrapper(traj.controls)) == trajectory_length

        # We iterate over two offsets to generate overlapping subsequences
        for offset in (0, subsequence_length // 2):

            def split_fn(x: np.ndarray) -> np.ndarray:
                """Private helper: splits arrays of arrays of shape `(T, ...)` into
                `(sections, subsequence_length, ...)`, where `sections = orig_length //
                subsequence_length`."""
                # Offset our starting point
                x = x[offset:]

                # Make sure our array is evenly divisible
                sections = len(x) // subsequence_length
                new_length = sections * subsequence_length
                x = x[:new_length]

                # Split & return
                return np.split(x, sections)

            for s, o, c in zip(
                split_fn(traj.states),
                fp.utils.SliceWrapper(
                    fp.utils.SliceWrapper(traj.observations).map(split_fn)
                ),
                fp.utils.SliceWrapper(
                    fp.utils.SliceWrapper(traj.controls).map(split_fn)
                ),
            ):
                # Add to subsequences
                subsequences.append(
                    types.TrajectoryNumpy(states=s, observations=o, controls=c)
                )
    return subsequences
