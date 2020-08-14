from typing import List

import pytest
import torch

import diffbayes
import fannypack
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
    state_dim,
)
from diffbayes import types


def _get_error(model) -> float:
    """Get the error of our models with a trainable output bias.

    Returns:
        float: Error. Compute as the absolute value of the output bias.
    """
    return abs(float(model.output_bias[0]))


def test_train_ekf_e2e(subsequence_dataloader, buddy):
    """Check that training our EKF end-to-end drops both dynamics and measurement
    errors.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearKalmanFilterMeasurementModel(trainable=True)
    initial_dynamics_error = _get_error(dynamics_model)
    initial_measurement_error = _get_error(measurement_model)

    filter_model = diffbayes.filters.ExtendedKalmanFilter(
        dynamics_model=dynamics_model, measurement_model=measurement_model
    )

    buddy.attach_model(filter_model)
    diffbayes.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )
    assert _get_error(dynamics_model) < initial_dynamics_error
    assert _get_error(measurement_model) < initial_measurement_error


def test_train_ukf_e2e(subsequence_dataloader, buddy):
    """Check that training our UKF end-to-end drops both dynamics and measurement
    errors.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearKalmanFilterMeasurementModel(trainable=True)
    initial_dynamics_error = _get_error(dynamics_model)
    initial_measurement_error = _get_error(measurement_model)

    filter_model = diffbayes.filters.UnscentedKalmanFilter(
        dynamics_model=dynamics_model, measurement_model=measurement_model
    )

    buddy.attach_model(filter_model)
    diffbayes.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )
    assert _get_error(dynamics_model) < initial_dynamics_error
    assert _get_error(measurement_model) < initial_measurement_error


def test_train_virtual_sensor_ekf_e2e(subsequence_dataloader, buddy):
    """Check that training our virtual sensor EKF end-to-end drops both dynamics and
    virtual sensor errors.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    virtual_sensor_model = LinearVirtualSensorModel(trainable=True)
    initial_dynamics_error = _get_error(dynamics_model)
    initial_virtual_sensor_error = _get_error(virtual_sensor_model)

    filter_model = diffbayes.filters.VirtualSensorExtendedKalmanFilter(
        dynamics_model=dynamics_model, virtual_sensor_model=virtual_sensor_model
    )

    buddy.attach_model(filter_model)
    diffbayes.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )
    assert _get_error(dynamics_model) < initial_dynamics_error
    assert _get_error(virtual_sensor_model) < initial_virtual_sensor_error


def test_train_pf_e2e(subsequence_dataloader, buddy):
    """Check that training our particle filter end-to-end drops both dynamics and
    measurement errors.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    measurement_model = LinearParticleFilterMeasurementModel(trainable=True)
    initial_dynamics_error = _get_error(dynamics_model)
    initial_measurement_error = _get_error(
        measurement_model.kalman_filter_measurement_model
    )

    filter_model = diffbayes.filters.ParticleFilter(
        dynamics_model=dynamics_model, measurement_model=measurement_model
    )

    buddy.attach_model(filter_model)
    diffbayes.train.train_filter(
        buddy,
        filter_model,
        subsequence_dataloader,
        initial_covariance=torch.eye(state_dim) * 0.01,
    )
    assert _get_error(dynamics_model) < initial_dynamics_error
    assert (
        _get_error(measurement_model.kalman_filter_measurement_model)
        < initial_measurement_error
    )
