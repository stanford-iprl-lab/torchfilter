"""Private module; avoid importing from directly.
"""

from typing import Optional, Tuple

import torch
from overrides import overrides

from .. import types, utils
from ..base import (
    DynamicsModel,
    KalmanFilterBase,
    KalmanFilterMeasurementModel,
    VirtualSensorModel,
)
from ._extended_kalman_filter import ExtendedKalmanFilter
from ._square_root_unscented_kalman_filter import SquareRootUnscentedKalmanFilter
from ._unscented_kalman_filter import UnscentedKalmanFilter


class _IdentityMeasurementModel(KalmanFilterMeasurementModel):
    """Identity measurement model. For use with our virtual sensor Kalman filters, which
    assume that the "observation" of the system is in the state space.

    Possible extension to consider in the future: we could very reasonably have both a
    virtual sensor _and_ a measurement model, which each map to a latent space.
    """

    def __init__(self, *, state_dim):
        self.scale_tril: types.ScaleTrilTorch
        """torch.Tensor: Lower-triangular uncertainty term, with shape
        `(N, state_dim, state_dim)`. This should be set externally."""

        super().__init__(state_dim=state_dim, observation_dim=state_dim)

    @overrides
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

        # Hacky: create a correctly-shaped scale_tril tensor
        states_N = states.shape[0]
        scale_tril_N = self.scale_tril.shape[0]
        if states_N == scale_tril_N:
            # Standard case: pass scale_tril computed from virtual sensor model as
            # measurement model noise
            scale_tril = self.scale_tril
        else:
            # For UKFs, our virtual sensor has a batch size of `N` but our measurement
            # model will get a batch size of `N * sigma_point_count`.  Here, we repeat
            # our noise values so that all sigma points within one "group" get the same
            # noise; this is inefficient and could be optimized in
            # the future.
            assert states_N % scale_tril_N == 0
            sigma_point_count = states_N // scale_tril_N
            assert sigma_point_count == 2 * self.state_dim + 1

            scale_tril = torch.repeat_interleave(
                self.scale_tril, sigma_point_count, dim=0
            )

        # Output
        virtual_observations = states
        return virtual_observations, scale_tril

    @overrides
    def jacobian(self, *, states: types.StatesTorch) -> torch.Tensor:
        """To avoid using autograd for computing our models Jacobian, we can directly
        return identity matrices.

        Args:
            states (torch.Tensor): Current state, size `(N, state_dim)`.

        Returns:
            torch.Tensor: Jacobian, size `(N, observation_dim, state_dim)`
        """
        N, state_dim = states.shape
        assert state_dim == self.state_dim == self.observation_dim
        return torch.eye(state_dim)[None, :, :].expand((N, state_dim, state_dim))


class _VirtualSensorKalmanFilterMixin(
    KalmanFilterBase  # Only needed for type-checking
):
    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        virtual_sensor_model: VirtualSensorModel,
        **kwargs,
    ):
        # Check submodule consistency
        assert isinstance(dynamics_model, DynamicsModel)

        # Initialize state dimension
        assert isinstance(self, KalmanFilterBase)
        super(_VirtualSensorKalmanFilterMixin, self).__init__(
            dynamics_model=dynamics_model,
            measurement_model=_IdentityMeasurementModel(
                state_dim=dynamics_model.state_dim
            ),
            **kwargs,
        )

        # Assign submodules
        self.virtual_sensor_model = virtual_sensor_model
        """diffbayes.base.VirtualSensorModel: Virtual sensor model."""

    @overrides
    def _update_step(self, *, observations: types.ObservationsTorch) -> None:
        (
            virtual_observations,
            self.measurement_model.scale_tril,
        ) = self.virtual_sensor_model(observations=observations)

        super()._update_step(observations=virtual_observations)

    def virtual_sensor_initialize_beliefs(
        self, *, observations: types.ObservationsTorch,
    ):
        """Use virtual sensor model to intialize filter beliefs.

        Args:
            observations (dict or torch.Tensor): observation inputs. should be
                either a dict of tensors or tensor of shape `(N, ...)`
        """
        mean, scale_tril = self.virtual_sensor_model(observations=observations)
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        self.initialize_beliefs(mean=mean, covariance=covariance)


class VirtualSensorExtendedKalmanFilter(
    _VirtualSensorKalmanFilterMixin, ExtendedKalmanFilter
):
    """EKF variant with a virtual sensor model for mapping raw observations to predicted
    states.

    Assumes measurement model is identity.
    """

    # Redefine constructor to remove **kwargs
    # This is for better static checking, makes language servers a little more useful
    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        virtual_sensor_model: VirtualSensorModel,
    ):
        super().__init__(
            dynamics_model=dynamics_model, virtual_sensor_model=virtual_sensor_model,
        )


class VirtualSensorUnscentedKalmanFilter(
    _VirtualSensorKalmanFilterMixin, UnscentedKalmanFilter
):
    """UKF variant with a virtual sensor model for mapping raw observations to predicted
    states.

    Assumes measurement model is identity.
    """

    # Redefine constructor to add explicit sigma_point_strategy kwarg
    # This is for better static checking, makes language servers a little more useful
    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        virtual_sensor_model: VirtualSensorModel,
        sigma_point_strategy: Optional[utils.SigmaPointStrategy] = None,
    ):
        super().__init__(
            dynamics_model=dynamics_model,
            virtual_sensor_model=virtual_sensor_model,
            sigma_point_strategy=sigma_point_strategy,
        )


class VirtualSensorSquareRootUnscentedKalmanFilter(
    _VirtualSensorKalmanFilterMixin, SquareRootUnscentedKalmanFilter
):
    """Square root UKF variant with a virtual sensor model for mapping raw observations
    to predicted states.

    Assumes measurement model is identity.
    """

    # Redefine constructor to add explicit sigma_point_strategy kwarg
    # This is for better static checking, makes language servers a little more useful
    def __init__(
        self,
        *,
        dynamics_model: DynamicsModel,
        virtual_sensor_model: VirtualSensorModel,
        sigma_point_strategy: Optional[utils.SigmaPointStrategy] = None,
    ):
        super().__init__(
            dynamics_model=dynamics_model,
            virtual_sensor_model=virtual_sensor_model,
            sigma_point_strategy=sigma_point_strategy,
        )
