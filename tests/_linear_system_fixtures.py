from typing import List, Tuple, cast

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
    timesteps = 2

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
    N = states.shape[0]

    output: List[types.TrajectoryNumpy] = []
    for i in range(N):
        output.append(
            types.TrajectoryNumpy(
                states=states[i, :, :],
                observations=observations[i, :, :],
                controls=controls[i, :, :],
            )
        )
    return output
