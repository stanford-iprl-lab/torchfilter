from typing import Tuple

import torch

import diffbayes
from diffbayes import types
from linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    LinearParticleFilterMeasurementModel,
    LinearVirtualSensorModel,
    generated_data,
    state_dim,
)


def test_particle_filter(generated_data):
    """Smoke test for particle filter.
    """
    _run_filter(
        diffbayes.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_extended_kalman_filter(generated_data):
    """Smoke test for EKF.
    """
    _run_filter(
        diffbayes.filters.ExtendedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_virtual_sensor_extended_kalman_filter(generated_data):
    """Smoke test for EKF w/ virtual sensor.
    """
    _run_filter(
        diffbayes.filters.VirtualSensorExtendedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            virtual_sensor_model=LinearVirtualSensorModel(),
        ),
        generated_data,
    )


def test_unscented_kalman_filter(generated_data):
    """Smoke test for UKF w/ Julier-style sigma points.
    """
    _run_filter(
        diffbayes.filters.UnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_unscented_kalman_filter_merwe(generated_data):
    """Smoke test for UKF w/ Merwe-style sigma points.
    """
    _run_filter(
        diffbayes.filters.UnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
            sigma_point_strategy=diffbayes.utils.MerweSigmaPointStrategy(
                dim=state_dim, alpha=1e-1
            ),
        ),
        generated_data,
    )


def test_virtual_sensor_ekf_consistency(generated_data):
    """Check that our Virtual Sensor EKF and standard EKF produce consistent results for
    a linear system.
    """
    # Create filters
    ekf = diffbayes.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    virtual_sensor_ekf = diffbayes.filters.VirtualSensorExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        virtual_sensor_model=LinearVirtualSensorModel(),
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(virtual_sensor_ekf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, virtual_sensor_ekf.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance, virtual_sensor_ekf.belief_covariance,
    )


def test_ukf_ekf_consistency(generated_data):
    """Check that our UKF and EKF produce consistent results for a linear system. (they
    should be identical)
    """
    # Create filters
    ekf = diffbayes.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    ukf = diffbayes.filters.UnscentedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(ukf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, ukf.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance, ukf.belief_covariance, rtol=1e-4, atol=5e-4
    )


def _run_filter(
    filter_model: diffbayes.base.Filter,
    data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ],
) -> torch.Tensor:
    states, observations, controls = data
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

    return estimated_states
