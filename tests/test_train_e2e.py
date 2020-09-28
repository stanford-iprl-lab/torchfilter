import torch
from _linear_system_fixtures import (
    buddy,
    generated_data,
    generated_data_numpy_list,
    single_step_dataloader,
    subsequence_dataloader,
)
from _linear_system_models import (
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    LinearParticleFilterMeasurementModel,
    LinearVirtualSensorModel,
    get_trainable_model_error,
    state_dim,
)

import torchfilter


def test_train_ekf_e2e(subsequence_dataloader, buddy):
    """Check that training our EKF end-to-end drops both dynamics and measurement
    errors.
    """
    # Create individual models + filter
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearKalmanFilterMeasurementModel(trainable=True)
    filter_model = torchfilter.filters.ExtendedKalmanFilter(
        dynamics_model=dynamics_model, measurement_model=measurement_model
    )

    # Compute initial errors
    initial_dynamics_error = get_trainable_model_error(dynamics_model)
    initial_measurement_error = get_trainable_model_error(measurement_model)

    # Train for 1 epoch
    buddy.attach_model(filter_model)
    torchfilter.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )

    # Check that errors dropped
    assert get_trainable_model_error(dynamics_model) < initial_dynamics_error
    assert get_trainable_model_error(measurement_model) < initial_measurement_error


def test_train_ukf_e2e(subsequence_dataloader, buddy):
    """Check that training our UKF end-to-end drops both dynamics and measurement
    errors.
    """
    # Create individual models + filter
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearKalmanFilterMeasurementModel(trainable=True)
    filter_model = torchfilter.filters.UnscentedKalmanFilter(
        dynamics_model=dynamics_model, measurement_model=measurement_model
    )

    # Compute initial errors
    initial_dynamics_error = get_trainable_model_error(dynamics_model)
    initial_measurement_error = get_trainable_model_error(measurement_model)

    # Train for 1 epoch
    buddy.attach_model(filter_model)
    torchfilter.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )

    # Check that errors dropped
    assert get_trainable_model_error(dynamics_model) < initial_dynamics_error
    assert get_trainable_model_error(measurement_model) < initial_measurement_error


def test_train_virtual_sensor_ekf_e2e(subsequence_dataloader, buddy):
    """Check that training our virtual sensor EKF end-to-end drops both dynamics and
    virtual sensor errors.
    """
    # Create individual models + filter
    dynamics_model = LinearDynamicsModel(trainable=True)
    virtual_sensor_model = LinearVirtualSensorModel(trainable=True)
    filter_model = torchfilter.filters.VirtualSensorExtendedKalmanFilter(
        dynamics_model=dynamics_model, virtual_sensor_model=virtual_sensor_model
    )

    # Compute initial errors
    initial_dynamics_error = get_trainable_model_error(dynamics_model)
    initial_virtual_sensor_error = get_trainable_model_error(virtual_sensor_model)

    # Train for 1 epoch
    buddy.attach_model(filter_model)
    torchfilter.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )

    # Check that errors dropped
    assert get_trainable_model_error(dynamics_model) < initial_dynamics_error
    assert (
        get_trainable_model_error(virtual_sensor_model) < initial_virtual_sensor_error
    )


def test_train_pf_e2e(subsequence_dataloader, buddy):
    """Check that training our particle filter end-to-end drops both dynamics and
    measurement errors.
    """
    # Create individual models + filter
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearParticleFilterMeasurementModel(trainable=True)
    filter_model = torchfilter.filters.ParticleFilter(
        dynamics_model=dynamics_model,
        measurement_model=measurement_model,
        num_particles=500,
    )

    # Compute initial errors
    initial_dynamics_error = get_trainable_model_error(dynamics_model)
    initial_measurement_error = get_trainable_model_error(
        measurement_model.kalman_filter_measurement_model
    )

    # Train for 1 epoch
    buddy.attach_model(filter_model)
    torchfilter.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )

    # Check that errors dropped
    assert get_trainable_model_error(dynamics_model) < initial_dynamics_error
    assert (
        get_trainable_model_error(measurement_model.kalman_filter_measurement_model)
        < initial_measurement_error
    )
