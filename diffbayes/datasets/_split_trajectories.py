import fannypack
import numpy as np


def split_trajectories(trajectories, subsequence_length):
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
        assert len(observations) == trajectory_length
        assert len(controls) == trajectory_length

        sections = trajectory_length // subsequence_length

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


def _split_helper(x, subsequence_length, offset):
    """Private helper: splits arrays or dicts of arrays of shape `(T, ...)`
    into `(sections, subsequence_length, ...)`.
    """
    if type(x) == np.ndarray:
        # Offset our starting point
        x = x[offset:]

        # Make sure our array is evenly divisible
        new_length = (len(x) // subsequence_length) * subsequence_length
        x = x[:new_length]

        # Split & return
        return np.split(x[:new_length], sections)
    elif type(x) == dict:
        # For dictionary inputs, we split the contents of each
        # value in the dictionary
        output = {}
        for key, value in x.items():
            output[key] = _split_helper(value, offset=offset)

        # Return a wrapped dictionary; this makes it iterable
        return fannypack.utils.SliceWrapper(output)
    else:
        assert False, "Invalid trajectory type"
