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
    state_dim,
)


def _get_error(model) -> float:
    """Get the error of our models with a trainable output bias.

    Returns:
        float: Error. Compute as the absolute value of the output bias.
    """
    return abs(float(model.output_bias[0]))


def test_train_dynamics_recurrent(subsequence_dataloader, buddy):
    """Check that our recurrent dynamics training drops model error.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    initial_error = _get_error(dynamics_model)

    buddy.attach_model(dynamics_model)
    diffbayes.train.train_dynamics_recurrent(
        buddy, dynamics_model, subsequence_dataloader
    )
    assert _get_error(dynamics_model) < initial_error


def test_train_dynamics_single_step(single_step_dataloader, buddy):
    """Check that our single-step dynamics training drops model error.
    """
    dynamics_model = LinearDynamicsModel(trainable=True)
    initial_error = _get_error(dynamics_model)

    buddy.attach_model(dynamics_model)
    diffbayes.train.train_dynamics_single_step(
        buddy, dynamics_model, single_step_dataloader
    )
    assert _get_error(dynamics_model) < initial_error


def test_train_virtual_sensor(single_step_dataloader, buddy):
    """Check that our virtual sensor training drops model error.
    """
    virtual_sensor_model = LinearVirtualSensorModel(trainable=True)
    initial_error = _get_error(virtual_sensor_model)

    buddy.attach_model(virtual_sensor_model)
    diffbayes.train.train_virtual_sensor_model(
        buddy, virtual_sensor_model, single_step_dataloader
    )
    assert _get_error(virtual_sensor_model) < initial_error


def test_train_particle_filter_measurement(
    particle_filter_measurement_dataloader, buddy
):
    """Check that our particle filter measurement training drops model error.
    """
    particle_filter_measurement_model = LinearParticleFilterMeasurementModel(
        trainable=True
    )
    initial_error = _get_error(
        particle_filter_measurement_model.kalman_filter_measurement_model
    )

    buddy.attach_model(particle_filter_measurement_model)
    diffbayes.train.train_particle_filter_measurement_model(
        buddy, particle_filter_measurement_model, particle_filter_measurement_dataloader
    )
    assert (
        _get_error(particle_filter_measurement_model.kalman_filter_measurement_model)
        < initial_error
    )
