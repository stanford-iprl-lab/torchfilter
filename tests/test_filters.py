from typing import Tuple

import torch

import diffbayes
from _linear_system_fixtures import generated_data
from _linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    LinearParticleFilterMeasurementModel,
    LinearVirtualSensorModel,
    state_dim,
)
from diffbayes import types


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


def test_particle_filter_resample(generated_data):
    """Smoke test for particle filter with resampling.
    """
    _run_filter(
        diffbayes.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
            resample=True,
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
            sigma_point_strategy=diffbayes.utils.MerweSigmaPointStrategy(alpha=1e-1),
        ),
        generated_data,
    )


def test_square_root_unscented_kalman_filter(generated_data):
    """Smoke test for SRUKF w/ Julier-style sigma points.
    """
    _run_filter(
        diffbayes.filters.SquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_square_root_unscented_kalman_filter_merwe(generated_data):
    """Smoke test for SRUKF w/ Merwe-style sigma points.
    """
    _run_filter(
        diffbayes.filters.SquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
            sigma_point_strategy=diffbayes.utils.MerweSigmaPointStrategy(alpha=1e-1),
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


def test_ukf_srukf_consistency(generated_data):
    """Check that our UKF and SRUKF produce consistent results for a linear system.
    (they should be identical)
    """
    # Create filters
    srukf = diffbayes.filters.SquareRootUnscentedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    ukf = diffbayes.filters.UnscentedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )

    # Run over data
    _run_filter(srukf, generated_data)
    _run_filter(ukf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(srukf.belief_mean, ukf.belief_mean)
    torch.testing.assert_allclose(
        srukf.belief_covariance, ukf.belief_covariance, rtol=1e-4, atol=5e-4
    )


def _run_filter(
    filter_model: diffbayes.base.Filter,
    data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ],
) -> torch.Tensor:
    """Helper for running a filter and returning estimated states.

    Args:
        filter_model (diffbayes.base.Filter): Filter to run.
        data (Tuple[
            types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
        ]): Data to run on. Shapes of all inputs should be `(T, N, *)`.

    Returns:
        torch.Tensor: Estimated states. Shape should be `(T - 1, N, state_dim)`.
    """

    # Get data
    states, observations, controls = data
    T, N, state_dim = states.shape

    # Initialize the filter belief to match the first timestep
    filter_model.initialize_beliefs(
        mean=states[0],
        covariance=torch.zeros(size=(N, state_dim, state_dim))
        + torch.eye(state_dim)[None, :, :] * 0.1,
    )

    # Run the filter on the remaining `T - 1` timesteps
    estimated_states = filter_model.forward_loop(
        observations=observations[1:], controls=controls[1:]
    )

    # Check output and return
    assert estimated_states.shape == (T - 1, N, state_dim)
    return estimated_states
