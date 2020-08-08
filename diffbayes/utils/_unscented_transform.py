from typing import Optional, Tuple, cast

import numpy as np
import torch

from .. import types


class UnscentedTransform:
    """ Helper class for performing (batched, differentiable) unscented transforms, with
    sigma point selection parameters described by [1].

    [1] http://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf

    Keyword Args:
        dim (int): Input dimension.
        alpha (float, optional): Spread parameter. Defaults to `1e-3`.
        kappa (float, optional): Secondary scaling parameter, which is typically set to
            `0.0` or `3 - dim`. Defaults to `0.0`.
        beta (float, optional): Extra sigma parameter. Defaults to `2` (optimal for
            Gaussians, as per [1]).
    """

    def __init__(
        self, *, dim: int, alpha=1e-3, kappa: float = 0.0, beta: float = 2.0,
    ):
        self._dim = dim

        # Sigma parameters
        self._lambd: float = (alpha ** 2) * (dim + kappa) - dim
        self._alpha: float = alpha
        self._beta: float = beta
        self._kappa: float = kappa

        # Sigma weights: we instantiate these lazily
        self._weights_c: Optional[torch.Tensor] = None
        self._weights_m: Optional[torch.Tensor] = None
        self._weights_device: Optional[torch.device] = None

    @property
    def weights_m(self) -> torch.Tensor:
        """Sigma point covariance weights. Instantiated on the first call to
        `compute_distribution()`."""
        assert self._weights_m is not None, "Weights not initialized!"
        return self._weights_m

    @property
    def weights_c(self) -> torch.Tensor:
        """Sigma point mean weights. Instantiated on the first call to
        `compute_distribution()`. Values should sum to `1`."""
        assert self._weights_c is not None, "Weights not initialized!"
        return self._weights_c

    def select_sigma_points(
        self, input_mean: torch.Tensor, input_covariance: types.CovarianceTorch,
    ) -> torch.Tensor:
        """Select sigma points.

        Args:
            input_mean (torch.Tensor): Distribution mean. Shape should be `(N, dim)`.
            input_covariance (torch.Tensor): Distribution covariance. Shape should be
                `(N, dim, dim)`.
        Returns:
            torch.Tensor: Selected sigma points, with shape `(N, 2 * dim + 1, dim)`.
        """

        N, dim = input_mean.shape
        assert input_covariance.shape == (N, dim, dim)
        assert dim == self._dim

        # Compute matrix root, offsets for sigma points
        #
        # Note that we offset with the row vectors, so we need an upper-triangular
        # cholesky decomposition [1].
        #
        # [1] https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf
        matrix_root = torch.cholesky((dim + self._lambd) * input_covariance, upper=True)
        assert matrix_root.shape == (N, dim, dim)

        sigma_point_offsets = input_mean.new_zeros((N, 2 * dim + 1, dim))
        sigma_point_offsets[:, 1 : 1 + dim] = matrix_root
        sigma_point_offsets[:, 1 + dim :] = -matrix_root

        # Create & return matrix of sigma points
        sigma_points: torch.Tensor = input_mean[:, None, :] + sigma_point_offsets
        assert sigma_points.shape == (N, 2 * dim + 1, dim)
        return sigma_points

    def compute_distribution(
        self, sigma_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, types.CovarianceTorch]:
        """Estimate a distribution from selected sigma points.

        Args:
            sigma_points (torch.Tensor): Sigma points, with shape
                `(N, 2 * dim + 1, dim)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and covariance, with shapes
            `(N, dim)` and `(N, dim, dim)` respectively.
        """
        # Lazy sigma weight initialization
        self._init_sigma_weights(sigma_points.device)
        weights_m = cast(torch.Tensor, self._weights_m)
        weights_c = cast(torch.Tensor, self._weights_c)

        # Check shapes
        N, sigma_point_count, dim = sigma_points.shape
        assert weights_m.shape == weights_c.shape == (sigma_point_count,)

        # Compute transformed mean, covariance
        transformed_mean = torch.sum(weights_m[None, :, None] * sigma_points, dim=1)
        assert transformed_mean.shape == (N, dim)

        sigma_points_centered = sigma_points - transformed_mean[:, None, :]
        transformed_covariance = torch.sum(
            weights_c[None, :, None, None]
            * (
                sigma_points_centered[:, :, :, None]
                @ sigma_points_centered[:, :, None, :]
            ),
            dim=1,
        )
        assert transformed_covariance.shape == (N, dim, dim)
        return transformed_mean, transformed_covariance

    def _init_sigma_weights(self, device: torch.device):
        """Helper for initializing sigma weights on a given device. Does nothing if
        weights already exist.

        Args:
            device (torch.device): Device to create weights on.
        """
        if self._weights_device == device:
            # Do nothing if weights already exist
            return

        self._weights_device = device
        self._weights_c = torch.full(
            size=(2 * self._dim + 1,),
            fill_value=1.0 / (2.0 * (self._dim + self._lambd)),
            device=device,
        )
        self._weights_m = self._weights_c.clone()
        self._weights_c[0] = self._lambd / (self._dim + self._lambd) + (
            1.0 - self._alpha ** 2 + self._beta
        )
        self._weights_m[0] = self._lambd / (self._dim + self._lambd)

        assert np.allclose(np.sum(self._weights_m.detach().numpy()), 1.0)
        assert not np.allclose(np.sum(self._weights_c.detach().numpy()), 1.0)
