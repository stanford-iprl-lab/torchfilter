from typing import Tuple

import torch

import diffbayes
from diffbayes import types
from linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    LinearParticleFilterMeasurementModel,
    generated_data,
)


def test_particle_filter(generated_data):
    _filter_smoke_test(
        diffbayes.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_extended_kalman_filter(generated_data):
    _filter_smoke_test(
        diffbayes.filters.ExtendedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_unscented_kalman_filter(generated_data):
    _filter_smoke_test(
        diffbayes.filters.UnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
            unscented_transform_params={"alpha": 0.2},
        ),
        generated_data,
    )


def _filter_smoke_test(
    filter_model: diffbayes.base.Filter,
    generated_data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ],
):
    states, observations, controls = generated_data
    timesteps, N, state_dim = states.shape

    filter_model.initialize_beliefs(
        mean=states[0],
        covariance=torch.zeros(size=(N, state_dim, state_dim))
        + torch.eye(state_dim)[None, :, :] * 0.1,
    )
    estimated_states = filter_model.forward_loop(
        observations=observations[1:], controls=controls[1:]
    )
    assert estimated_states.shape == (timesteps - 1, N, state_dim)
