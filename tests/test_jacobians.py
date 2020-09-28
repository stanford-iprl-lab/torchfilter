import torch
from _linear_system_models import (
    A,
    C,
    LinearDynamicsModel,
    LinearKalmanFilterMeasurementModel,
    control_dim,
    state_dim,
)


def test_dynamics_jacobian():
    """Checks that our autograd-computed dynamics jacobian is correct."""
    N = 10
    dynamics_model = LinearDynamicsModel()
    A_autograd = dynamics_model.jacobian(
        initial_states=torch.zeros((N, state_dim)),
        controls=torch.zeros((N, control_dim)),
    )

    for i in range(N):
        torch.testing.assert_allclose(A_autograd[i], A)


def test_measurement_jacobian():
    """Checks that our autograd-computed measurement jacobian is correct."""
    N = 10
    measurement_model = LinearKalmanFilterMeasurementModel()
    C_autograd = measurement_model.jacobian(states=torch.zeros((N, state_dim)))

    for i in range(N):
        torch.testing.assert_allclose(C_autograd[i], C)
