import os
import shutil
from typing import List, Tuple

import numpy as np
import pytest
import torch

import diffbayes
import fannypack as fp
from _linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    control_dim,
    observation_dim,
    state_dim,
)
from diffbayes import types


@pytest.fixture
def generated_data() -> Tuple[
    types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
]:
    """Generate `N` (noisy) trajectories using our dynamics and measurement models.

    Returns:
        tuple: (states, observations, controls). First dimension of all tensors should
            be `N`.
    """
    torch.random.manual_seed(0)
    N = 5
    timesteps = 100

    dynamics_model = LinearDynamicsModel()
    measurement_model = LinearKalmanFilterMeasurementModel()

    # Initialize empty states, observations
    states = torch.zeros((timesteps, N, state_dim))
    observations = torch.zeros((timesteps, N, observation_dim))

    # Generate random control inputs
    controls = torch.randn(size=(timesteps, N, control_dim))

    for t in range(timesteps):
        if t == 0:
            # Initialize random initial state
            states[0, :, :] = torch.randn(size=(N, state_dim))
        else:
            # Update state and add noise
            pred_states, Q_tril = dynamics_model(
                initial_states=states[t - 1, :, :], controls=controls[t, :, :]
            )
            assert pred_states.shape == (N, state_dim)
            assert Q_tril.shape == (N, state_dim, state_dim)

            states[t, :, :] = pred_states + (
                Q_tril @ torch.randn(size=(N, state_dim, 1))
            ).squeeze(-1)

        # Compute observations and add noise
        pred_observations, R_tril = measurement_model(states=states[t, :, :])
        observations[t, :, :] = pred_observations + (
            R_tril @ torch.randn(size=(N, observation_dim, 1))
        ).squeeze(-1)

    return states, observations, controls


@pytest.fixture
def generated_data_numpy_list(
    generated_data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ]
) -> List[types.TrajectoryNumpy]:
    """Same as `generated_data()`, but returns a list of individual trajectories instead
    of batched trajectories. This is closer to what real-world datasets often look like,
    as each trajectory can now be a different length.

    Returns:
        List[types.TrajectoryNumpy]: List of trajectories.
    """
    states, observations, controls = fp.utils.to_numpy(generated_data)
    N = states.shape[1]

    output: List[types.TrajectoryNumpy] = []
    for i in range(N):
        output.append(
            types.TrajectoryNumpy(
                states=states[:, i, :],
                observations=observations[:, i, :],
                controls=controls[:, i, :],
            )
        )
    return output


@pytest.fixture
def subsequence_dataloader(
    generated_data_numpy_list: List[types.TrajectoryNumpy],
) -> torch.utils.data.DataLoader:
    dataset = diffbayes.data.SubsequenceDataset(
        generated_data_numpy_list, subsequence_length=10
    )
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def single_step_dataloader(
    generated_data_numpy_list: List[types.TrajectoryNumpy],
) -> torch.utils.data.DataLoader:
    dataset = diffbayes.data.SingleStepDataset(generated_data_numpy_list)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture
def particle_filter_measurement_dataloader(
    generated_data_numpy_list: List[types.TrajectoryNumpy],
) -> torch.utils.data.DataLoader:
    dataset = diffbayes.data.ParticleFilterMeasurementDataset(
        generated_data_numpy_list,
        covariance=np.identity(state_dim) * 0.05,
        samples_per_pair=10,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=16)


@pytest.fixture()
def buddy():
    """Fixture for setting up a Buddy, as well as some dummy training data.
    """
    # Construct and yield a training buddy
    yield fp.utils.Buddy(
        "temporary_buddy",
        # Use directories relative to this fixture
        checkpoint_dir=os.path.join(
            os.path.dirname(__file__), "tmp/assets/checkpoints/"
        ),
        metadata_dir=os.path.join(os.path.dirname(__file__), "tmp/assets/metadata/"),
        log_dir=os.path.join(os.path.dirname(__file__), "tmp/assets/logs/"),
        verbose=True,
        # Disable auto-checkpointing
        optimizer_checkpoint_interval=0,
        cpu_only=True,
    )

    # Delete log files, metadata, checkpoints, etc when done
    path = os.path.join(os.path.dirname(__file__), "tmp/")
    if os.path.isdir(path):
        shutil.rmtree(path)
