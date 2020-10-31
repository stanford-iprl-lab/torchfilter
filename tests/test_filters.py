from typing import Tuple

import torch
from _linear_system_fixtures import generated_data
from _linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    LinearParticleFilterMeasurementModel,
    LinearVirtualSensorModel,
    state_dim,
)

import torchfilter
from torchfilter import types


def test_particle_filter(generated_data):
    """Smoke test for particle filter."""
    _run_filter(
        torchfilter.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_particle_filter_resample(generated_data):
    """Smoke test for particle filter with resampling."""
    _run_filter(
        torchfilter.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
            resample=True,
        ),
        generated_data,
    )


def test_particle_filter_soft_resample(generated_data):
    """Smoke test for particle filter with soft-resampling."""
    _run_filter(
        torchfilter.filters.ParticleFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearParticleFilterMeasurementModel(),
            resample=True,
            soft_resample_alpha=0.5,
        ),
        generated_data,
    )


def test_particle_filter_dynamic_particle_count(generated_data):
    """Smoke test for particle filter with a dynamically changing particle count + no resampling."""
    filter_model = torchfilter.filters.ParticleFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearParticleFilterMeasurementModel(),
        resample=False,
        num_particles=30,
    )
    _run_filter(filter_model, generated_data)
    assert filter_model.particle_states.shape[1] == 30

    # Expand
    filter_model.num_particles = 100
    _run_filter(filter_model, generated_data, initialize_beliefs=False)
    assert filter_model.particle_states.shape[1] == 100

    # Contract
    filter_model.num_particles = 30
    _run_filter(filter_model, generated_data, initialize_beliefs=False)
    assert filter_model.particle_states.shape[1] == 30


def test_particle_filter_dynamic_particle_count_resample(generated_data):
    """Smoke test for particle filter with a dynamically changing particle count w/ resampling."""
    filter_model = torchfilter.filters.ParticleFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearParticleFilterMeasurementModel(),
        resample=True,
        num_particles=30,
    )
    _run_filter(filter_model, generated_data)
    assert filter_model.particle_states.shape[1] == 30

    # Expand
    filter_model.num_particles = 100
    _run_filter(filter_model, generated_data, initialize_beliefs=False)
    assert filter_model.particle_states.shape[1] == 100

    # Contract
    filter_model.num_particles = 30
    _run_filter(filter_model, generated_data, initialize_beliefs=False)
    assert filter_model.particle_states.shape[1] == 30


def test_ekf(generated_data):
    """Smoke test for EKF."""
    _run_filter(
        torchfilter.filters.ExtendedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_virtual_sensor_ekf(generated_data):
    """Smoke test for EKF w/ virtual sensor."""
    _run_filter(
        torchfilter.filters.VirtualSensorExtendedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            virtual_sensor_model=LinearVirtualSensorModel(),
        ),
        generated_data,
    )


def test_eif(generated_data):
    """Smoke test for EIF."""
    _run_filter(
        torchfilter.filters.ExtendedInformationFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_ukf(generated_data):
    """Smoke test for UKF w/ Julier-style sigma points."""
    _run_filter(
        torchfilter.filters.UnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_ukf_merwe(generated_data):
    """Smoke test for UKF w/ Merwe-style sigma points."""
    _run_filter(
        torchfilter.filters.UnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
            sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=1e-1),
        ),
        generated_data,
    )


def test_virtual_sensor_ukf(generated_data):
    """Smoke test for virtual sensor UKF w/ Julier-style sigma points."""
    _run_filter(
        torchfilter.filters.VirtualSensorUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            virtual_sensor_model=LinearVirtualSensorModel(),
            sigma_point_strategy=torchfilter.utils.JulierSigmaPointStrategy(),  # optional
        ),
        generated_data,
    )


def test_srukf(generated_data):
    """Smoke test for SRUKF w/ Julier-style sigma points."""
    _run_filter(
        torchfilter.filters.SquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
        ),
        generated_data,
    )


def test_srukf_merwe(generated_data):
    """Smoke test for SRUKF w/ Merwe-style sigma points."""
    _run_filter(
        torchfilter.filters.SquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            measurement_model=LinearKalmanFilterMeasurementModel(),
            sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=1e-1),
        ),
        generated_data,
    )


def test_virtual_sensor_srukf(generated_data):
    """Smoke test for virtual sensor SRUKF w/ Julier-style sigma points."""
    _run_filter(
        torchfilter.filters.VirtualSensorSquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            virtual_sensor_model=LinearVirtualSensorModel(),
            sigma_point_strategy=torchfilter.utils.JulierSigmaPointStrategy(),  # optional
        ),
        generated_data,
    )


def test_virtual_sensor_ekf_consistency(generated_data):
    """Check that our Virtual Sensor EKF and standard EKF produce consistent results for
    a linear system.
    """
    # Create filters
    ekf = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    virtual_sensor_ekf = torchfilter.filters.VirtualSensorExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        virtual_sensor_model=LinearVirtualSensorModel(),
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(virtual_sensor_ekf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, virtual_sensor_ekf.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance,
        virtual_sensor_ekf.belief_covariance,
    )


def test_virtual_sensor_ukf_consistency(generated_data):
    """Check that our Virtual Sensor UKF and standard EKF produce consistent results for
    a linear system.
    """
    # Create filters
    ekf = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    virtual_sensor_ukf = torchfilter.filters.VirtualSensorUnscentedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        virtual_sensor_model=LinearVirtualSensorModel(),
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(virtual_sensor_ukf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, virtual_sensor_ukf.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance,
        virtual_sensor_ukf.belief_covariance,
        rtol=1e-4,
        atol=5e-4,
    )


def test_virtual_sensor_srukf_consistency(generated_data):
    """Check that our Virtual Sensor SRUKF and standard EKF produce consistent results for
    a linear system.
    """
    # Create filters
    ekf = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    virtual_sensor_srukf = (
        torchfilter.filters.VirtualSensorSquareRootUnscentedKalmanFilter(
            dynamics_model=LinearDynamicsModel(),
            virtual_sensor_model=LinearVirtualSensorModel(),
        )
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(virtual_sensor_srukf, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, virtual_sensor_srukf.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance,
        virtual_sensor_srukf.belief_covariance,
        rtol=1e-4,
        atol=5e-4,
    )


def test_eif_ekf_consistency(generated_data):
    """Check that our EIF and EKF produce consistent results for a linear system. (they
    should be identical)
    """
    # Create filters
    ekf = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    eif = torchfilter.filters.ExtendedInformationFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )

    # Run over data
    _run_filter(ekf, generated_data)
    _run_filter(eif, generated_data)

    # Check final beliefs
    torch.testing.assert_allclose(ekf.belief_mean, eif.belief_mean)
    torch.testing.assert_allclose(
        ekf.belief_covariance, eif.belief_covariance, rtol=1e-4, atol=5e-4
    )


def test_ukf_ekf_consistency(generated_data):
    """Check that our UKF and EKF produce consistent results for a linear system. (they
    should be identical)
    """
    # Create filters
    ekf = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    ukf = torchfilter.filters.UnscentedKalmanFilter(
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
    srukf = torchfilter.filters.SquareRootUnscentedKalmanFilter(
        dynamics_model=LinearDynamicsModel(),
        measurement_model=LinearKalmanFilterMeasurementModel(),
    )
    ukf = torchfilter.filters.UnscentedKalmanFilter(
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
    filter_model: torchfilter.base.Filter,
    data: Tuple[
        types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
    ],
    initialize_beliefs: bool = True,
) -> torch.Tensor:
    """Helper for running a filter and returning estimated states.

    Args:
        filter_model (torchfilter.base.Filter): Filter to run.
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
    if initialize_beliefs:
        filter_model.initialize_beliefs(
            mean=states[0],
            covariance=torch.zeros(size=(N, state_dim, state_dim))
            + torch.eye(state_dim)[None, :, :] * 0.5,
        )

    # Run the filter on the remaining `T - 1` timesteps
    estimated_states = filter_model.forward_loop(
        observations=observations[1:], controls=controls[1:]
    )

    # Check output and return
    assert estimated_states.shape == (T - 1, N, state_dim)
    return estimated_states
