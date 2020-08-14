from typing import Tuple, cast

import torch
import torch.nn as nn

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
    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=state_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([1.0]))

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

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, Q_tril[None, :, :].expand((N, state_dim, state_dim))


class LinearKalmanFilterMeasurementModel(diffbayes.base.KalmanFilterMeasurementModel):
    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=state_dim, observation_dim=observation_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([1.0]))

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

        # Compute output
        observations = (C[None, :, :] @ states[:, :, None]).squeeze(-1)
        scale_tril = R_tril[None, :, :].expand((N, observation_dim, observation_dim))

        # Add output bias if trainable
        if self.trainable:
            observations += self.output_bias

        # Compute/return predicted measurement and noise values
        return observations, scale_tril


class LinearVirtualSensorModel(diffbayes.base.VirtualSensorModel):
    def __init__(self, trainable: bool = False):
        super().__init__(state_dim=state_dim)

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([1.0]))

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
        # Note that for square C_pinv matrices, we can compute scale_tril as C_pinv @
        # R_tril. In the general case, we transform the full covariance and then take
        # the cholesky decomposition.
        predicted_states = (C_pinv[None, :, :] @ observations[:, :, None]).squeeze(-1)
        scale_tril = torch.cholesky(
            C_pinv @ R_tril @ R_tril.transpose(-1, -2) @ C_pinv.transpose(-1, -2)
        )[None, :, :].expand((N, state_dim, state_dim))

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, scale_tril


class LinearParticleFilterMeasurementModel(
    diffbayes.base.WrappedParticleFilterMeasurementModel
):
    def __init__(self, trainable: bool = False):
        super().__init__(
            kalman_filter_measurement_model=LinearKalmanFilterMeasurementModel(
                trainable=trainable
            )
        )
