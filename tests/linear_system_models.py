from typing import Tuple, cast

import pytest
import torch

import diffbayes
from diffbayes import types

state_dim = 10
control_dim = 3
observation_dim = 3

A = torch.empty(size=(state_dim, state_dim))
torch.nn.init.orthogonal_(A, gain=1.0)

B = torch.randn(size=(state_dim, control_dim))
C = torch.randn(size=(observation_dim, state_dim))
Q_tril = torch.eye(state_dim) * 0.1
R_tril = torch.eye(observation_dim) * 0.1


class LinearDynamicsModel(diffbayes.base.DynamicsModel):
    def __init__(self):
        super().__init__(state_dim=state_dim)

    def forward(
        self, *, initial_states: types.StatesTorch, controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Forward step for a discrete linear dynamical system.

        Args:
            initial_states (torch.Tensor): Initial states of our system.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """

        # Controls should be tensor, not dictionary
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)

        # Check shapes
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert A.shape == (state_dim, state_dim)
        assert N == N_alt

        # Compute/return states and noise values
        predicted_states = (A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
            B[None, :, :] @ controls[:, :, None]
        ).squeeze(-1)
        return predicted_states, Q_tril[None, :, :].expand((N, state_dim, state_dim))


class LinearKalmanFilterMeasurementModel(diffbayes.base.KalmanFilterMeasurementModel):
    def __init__(self):
        super().__init__(state_dim=state_dim, observation_dim=observation_dim)

    def forward(
        self, *, states: types.StatesTorch
    ) -> Tuple[types.ObservationsNoDictTorch, types.ScaleTrilTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected observations
            and cholesky decomposition of covariance.  Shape should be `(N, M)`.
        """
        # Check shape
        N = states.shape[0]
        assert states.shape == (N, state_dim)

        # Compute/return predicted measurement and noise values
        return (
            (C[None, :, :] @ states[:, :, None]).squeeze(-1),
            R_tril[None, :, :].expand((N, observation_dim, observation_dim)),
        )


class LinearParticleFilterMeasurementModel(
    diffbayes.base.WrappedParticleFilterMeasurementModel
):
    def __init__(self):
        super().__init__(
            kalman_filter_measurement_model=LinearKalmanFilterMeasurementModel()
        )


@pytest.fixture
def generated_data() -> Tuple[
    types.StatesTorch, types.ObservationsNoDictTorch, types.ControlsNoDictTorch
]:
    N = 2
    timesteps = 200

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
