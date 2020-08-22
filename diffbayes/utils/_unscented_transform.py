import warnings
from typing import Optional, Tuple

import numpy as np
import torch

import fannypack

from .. import types
from ._sigma_points import JulierSigmaPointStrategy, SigmaPointStrategy


class UnscentedTransform:
    """Helper class for performing (batched, differentiable) unscented transforms, with
    sigma point selection parameters described by [1].

    [1] http://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf

    Keyword Args:
        dim (int): Input dimension.
        sigma_point_strategy (diffbayes.utils.SigmaPointStrategy, optional): Strategy to
            use for sigma point selection. Defaults to Julier.
    """

    def __init__(
        self,
        *,
        dim: int,
        sigma_point_strategy: SigmaPointStrategy = JulierSigmaPointStrategy(),
    ):
        # Sigma weights
        weights_c, weights_m = sigma_point_strategy.compute_sigma_weights(dim=dim)

        self.weights_c: torch.Tensor = weights_c
        """torch.Tensor: Unscented transform covariance weights. Note that this will be
        initially instantiated on the CPU, and moved in `compute_distribution()`."""
        self.weights_m: torch.Tensor = weights_m
        """torch.Tensor: Unscented transform mean weights. Note that this will be
        initially instantiated on the CPU, and moved in `compute_distribution()`."""

        # State dimensionality
        self._dim = dim

        # Sigma point spread parameter
        self._lambd = sigma_point_strategy.compute_lambda(dim=dim)

        if self._lambd + dim < 1e-3:
            warnings.warn(
                "Unscented transform parameters may result in a very small matrix root;"
                " consider tuning.",
                RuntimeWarning,
                stacklevel=1,
            )

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
        return self.select_sigma_points_square_root(
            input_mean, input_scale_tril=torch.cholesky(input_covariance)
        )

    def select_sigma_points_square_root(
        self, input_mean: torch.Tensor, input_scale_tril: types.ScaleTrilTorch,
    ) -> torch.Tensor:
        """Select sigma points using square root of covariance.

        Args:
            input_mean (torch.Tensor): Distribution mean. Shape should be `(N, dim)`.
            input_scale_tril (torch.Tensor): Cholesky decomposition of distribution
                covariance. Shape should be `(N, dim, dim)`.
        Returns:
            torch.Tensor: Selected sigma points, with shape `(N, 2 * dim + 1, dim)`.
        """

        N, dim = input_mean.shape
        assert input_scale_tril.shape == (N, dim, dim)
        assert dim == self._dim

        # Compute matrix root, offsets for sigma points
        #
        # Note that we offset with the row vectors, so we need an upper-triangular
        # cholesky decomposition [1].
        #
        # [1] https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf
        matrix_root = np.sqrt(dim + self._lambd) * input_scale_tril.transpose(-1, -2)
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
        # Make sure devices match
        device = sigma_points.device
        if self.weights_c.device != device:
            self.weights_c = self.weights_c.to(device)
            self.weights_m = self.weights_m.to(device)

        # Check shapes
        N, sigma_point_count, dim = sigma_points.shape
        assert self.weights_m.shape == self.weights_c.shape == (sigma_point_count,)

        # Compute transformed mean, covariance
        transformed_mean = torch.sum(
            self.weights_m[None, :, None] * sigma_points, dim=1
        )
        assert transformed_mean.shape == (N, dim)

        sigma_points_centered = sigma_points - transformed_mean[:, None, :]
        # sigma_points_centered = sigma_points - sigma_points[:, 0:1, :]
        transformed_covariance = torch.sum(
            self.weights_c[None, :, None, None]
            * (
                sigma_points_centered[:, :, :, None]
                @ sigma_points_centered[:, :, None, :]
            ),
            dim=1,
        )
        assert transformed_covariance.shape == (N, dim, dim)
        return transformed_mean, transformed_covariance

    def compute_distribution_square_root(
        self,
        sigma_points: torch.Tensor,
        additive_noise_scale_tril: Optional[types.ScaleTrilTorch] = None,
    ) -> Tuple[torch.Tensor, types.ScaleTrilTorch]:
        """Estimate a distribution from selected sigma points; square root formulation.

        Args:
            sigma_points (torch.Tensor): Sigma points, with shape
                `(N, 2 * dim + 1, dim)`.
            additive_noise_scale_tril (torch.Tensor, optional): Parameterizes an
                additive Gaussian noise term. Should be lower-trinagular, with shape
                `(N, dim, dim)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and square root of covariance, with
            shapes `(N, dim)` and `(N, dim, dim)` respectively.
        """
        # Make sure devices match
        device = sigma_points.device
        if self.weights_c.device != device:
            self.weights_c = self.weights_c.to(device)
            self.weights_m = self.weights_m.to(device)

        # Check shapes
        N, sigma_point_count, dim = sigma_points.shape
        assert self.weights_m.shape == self.weights_c.shape == (sigma_point_count,)

        # Default: no additive noise
        if additive_noise_scale_tril is None:
            additive_noise_scale_tril = sigma_points.new_zeros((N, dim, dim))
        else:
            assert additive_noise_scale_tril.shape == (N, dim, dim)

        # Compute transformed mean, covariance
        transformed_mean = torch.sum(
            self.weights_m[None, :, None] * sigma_points, dim=1
        )
        assert transformed_mean.shape == (N, dim)

        sigma_points_centered = sigma_points - transformed_mean[:, None, :]
        # sigma_points_centered = sigma_points - sigma_points[:, 0:1, :]

        concatenated = torch.cat(
            [
                sigma_points_centered[:, 1:, :] * torch.sqrt(self.weights_c[1]),
                additive_noise_scale_tril.transpose(-1, -2),
            ],
            dim=1,
        )
        _unused_Q, R = torch.qr(concatenated, some=False)
        L = R[:, :dim, :].transpose(-1, -2)
        assert L.shape == (N, dim, dim)

        transformed_scale_tril = fannypack.utils.cholupdate(
            L=L, x=sigma_points_centered[:, 0, :], weight=self.weights_c[0],
        )

        assert transformed_scale_tril.shape == (N, dim, dim)
        return transformed_mean, transformed_scale_tril
