import diffbayes
import fannypack
from _linear_system_fixtures import (
    buddy,
    generated_data,
    generated_data_numpy_list,
    particle_filter_measurement_dataloader,
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


def test_train_dynamics_recurrent(subsequence_dataloader, buddy):
    """Check that our recurrent dynamics training drops model error.
    """
    # Create individual model
    dynamics_model = LinearDynamicsModel(trainable=True)

    # Compute initial error
    initial_error = get_trainable_model_error(dynamics_model)

    # Train for 1 epoch
    buddy.attach_model(dynamics_model)
    diffbayes.train.train_dynamics_recurrent(
        buddy, dynamics_model, subsequence_dataloader
    )

    # Check that error dropped
    assert get_trainable_model_error(dynamics_model) < initial_error


def test_train_dynamics_single_step(single_step_dataloader, buddy):
    """Check that our single-step dynamics training drops model error.
    """
    # Create individual model
    dynamics_model = LinearDynamicsModel(trainable=True)

    # Compute initial error
    initial_error = get_trainable_model_error(dynamics_model)

    # Train for 1 epoch
    buddy.attach_model(dynamics_model)
    diffbayes.train.train_dynamics_single_step(
        buddy, dynamics_model, single_step_dataloader
    )

    # Check that error dropped
    assert get_trainable_model_error(dynamics_model) < initial_error


def test_train_virtual_sensor(single_step_dataloader, buddy):
    """Check that our virtual sensor training drops model error.
    """
    # Create individual model
    virtual_sensor_model = LinearVirtualSensorModel(trainable=True)

    # Compute initial error
    initial_error = get_trainable_model_error(virtual_sensor_model)

    # Train for 1 epoch
    buddy.attach_model(virtual_sensor_model)
    diffbayes.train.train_virtual_sensor(
        buddy, virtual_sensor_model, single_step_dataloader
    )

    # Check that error dropped
    assert get_trainable_model_error(virtual_sensor_model) < initial_error


def test_train_kalman_filter_measurement(single_step_dataloader, buddy):
    """Check that our Kalman filter measurement training drops model error.
    """
    # Create individual model
    measurement_model = LinearKalmanFilterMeasurementModel(trainable=True)

    # Compute initial error
    initial_error = get_trainable_model_error(measurement_model)

    # Train for 1 epoch
    buddy.attach_model(measurement_model)
    diffbayes.train.train_kalman_filter_measurement(
        buddy, measurement_model, single_step_dataloader
    )

    # Check that error dropped
    assert get_trainable_model_error(measurement_model) < initial_error


def test_train_particle_filter_measurement(
    particle_filter_measurement_dataloader, buddy
):
    """Check that our particle filter measurement training drops model error.
    """
    # Create individual model
    particle_filter_measurement_model = LinearParticleFilterMeasurementModel(
        trainable=True
    )

    # Compute initial error
    initial_error = get_trainable_model_error(
        particle_filter_measurement_model.kalman_filter_measurement_model
    )

    # Train for 1 epoch
    buddy.attach_model(particle_filter_measurement_model)
    diffbayes.train.train_particle_filter_measurement(
        buddy, particle_filter_measurement_model, particle_filter_measurement_dataloader
    )

    # Check that error dropped
    assert (
        get_trainable_model_error(
            particle_filter_measurement_model.kalman_filter_measurement_model
        )
        < initial_error
    )
