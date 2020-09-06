from typing import List

import numpy as np

import torchfilter


def test_split_trajectories():
    """Basic check for subsequence generation helper.
    """

    num_trajectories = 100
    trajectory_timesteps = 100
    subsequence_length = 10

    # Generate trajectories
    trajectories: List[torchfilter.types.TrajectoryNumpy] = []
    for i in range(num_trajectories):
        trajectories.append(
            torchfilter.types.TrajectoryNumpy(
                states=np.ones((trajectory_timesteps, 5)),
                observations={
                    "key": np.zeros((trajectory_timesteps, 5)),
                    "some_other_key": np.zeros((trajectory_timesteps, 5)),
                },
                controls=np.zeros((trajectory_timesteps, 5)),
            )
        )

    # Split into subsequences
    subsequences = torchfilter.data.split_trajectories(
        trajectories, subsequence_length=subsequence_length
    )

    # Validate subsequences
    assert len(subsequences) == num_trajectories * 19
    for traj in subsequences:
        assert traj.states.shape == (subsequence_length, 5)
        assert traj.controls.shape == (subsequence_length, 5)
        assert np.allclose(traj.states, 1.0)
        assert "key" in traj.observations
        assert "some_other_key" in traj.observations
        assert np.allclose(traj.controls, 0.0)
