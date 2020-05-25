from typing import List, cast

import numpy as np

import fannypack

from .. import types


def split_trajectories(
    trajectories: List[types.TrajectoryTupleNumpy], subsequence_length: int
) -> List[types.TrajectoryTupleNumpy]:
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
        trajectories (list): list of trajectories, where each is a tuple of
            `(states, observations, controls)`. Each tuple member should be
            either a numpy array or dict of numpy arrays with shape `(T, ...)`.
        subsequence_length (int): # of timesteps per subsequence.
    Returns:
        list: A list of subsequences, as `(states, observations, controls)`
        tuples. Each tuple member should be either a numpy array or dict of
        numpy arrays with shape `(subsequence_length, ...)`.
    """

    subsequences = []

    for trajectory in trajectories:
        # Chop up each trajectory into overlapping subsequences
        assert len(trajectory) == 3
        states, observations, controls = trajectory

        trajectory_length = len(states)
        assert len(fannypack.utils.SliceWrapper(observations)) == trajectory_length
        assert len(fannypack.utils.SliceWrapper(controls)) == trajectory_length

        # We iterate over two offsets to generate overlapping subsequences
        for offset in (0, subsequence_length // 2):
            for s, o, c in zip(
                _split_helper(states, subsequence_length, offset),
                _split_helper(observations, subsequence_length, offset),
                _split_helper(controls, subsequence_length, offset),
            ):
                # Numpy => Torch
                s = fannypack.utils.to_torch(s)
                o = fannypack.utils.to_torch(o)
                c = fannypack.utils.to_torch(c)

                # Add to subsequences
                subsequences.append((s, o, c))
    return subsequences


def _split_helper(
    x: types.NumpyArrayOrDict, subsequence_length: int, offset: int,
) -> types.NumpyArrayOrDict:
    """Private helper: splits arrays or dicts of arrays of shape `(T, ...)`
    into `(sections, subsequence_length, ...)`, where `sections = orig_length //
    subsequence_length`.
    """
    if type(x) == np.ndarray:
        x = cast(np.ndarray, x)

        # Offset our starting point
        x = x[offset:]

        # Make sure our array is evenly divisible
        sections = len(x) // subsequence_length
        new_length = sections * subsequence_length
        x = x[:new_length]

        # Split & return
        return np.split(x, sections)
    elif type(x) == dict:
        # For dictionary inputs, we split the contents of each
        # value in the dictionary
        output = {}
        for key, value in x.items():
            output[key] = _split_helper(value, subsequence_length, offset)

        # Return a wrapped dictionary; this makes it iterable
        return fannypack.utils.SliceWrapper(output)
    else:
        assert False, "Invalid trajectory type"
