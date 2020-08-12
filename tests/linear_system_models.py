from typing import Tuple, cast

import pytest
import torch

import diffbayes
from diffbayes import types

state_dim = 5
control_dim = 3
observation_dim = 7

torch.random.manual_seed(0)
A = torch.empty(size=(state_dim, state_dim))
torch.nn.init.orthogonal_(A, gain=1.0)

B = torch.randn(size=(state_dim, control_dim))
C = torch.randn(size=(observation_dim, state_dim))
C_pinv = torch.pinverse(C)
Q_tril = torch.eye(state_dim) * 0.02
R_tril = torch.eye(observation_dim) * 0.05


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


class LinearVirtualSensorModel(diffbayes.base.VirtualSensorModel):
    def __init__(self):
        super().__init__(state_dim=state_dim)

    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Predicts states and uncertainties from observation inputs.

        Uncertainties should be lower-triangular Cholesky decompositions of covariance
        matrices.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """
        # Observations should be tensor, not dictionary
        assert isinstance(observations, torch.Tensor)
        observations = cast(torch.Tensor, observations)
        N = observations.shape[0]

        # Compute/return predicted state and uncertainty values
        # Note that for square C_pinv matrices, we can compute scale_tril as simply
        # C_pinv @ R_tril. In the general case, we transform the full covariance and
        # then take the cholesky decomposition.
        predicted_states = (C_pinv[None, :, :] @ observations[:, :, None]).squeeze(-1)
        scale_tril = torch.cholesky(
            C_pinv @ R_tril @ R_tril.transpose(-1, -2) @ C_pinv.transpose(-1, -2)
        )[None, :, :].expand((N, state_dim, state_dim))
        return predicted_states, scale_tril


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
    torch.random.manual_seed(0)
    N = 5
    timesteps = 2

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
