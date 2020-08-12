import warnings
from typing import Optional, Tuple

import torch

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
        self, *, dim: int, sigma_point_strategy: Optional[SigmaPointStrategy] = None,
    ):
        self._dim = dim

        if sigma_point_strategy is None:
            sigma_point_strategy = JulierSigmaPointStrategy(dim=dim)
        self.sigma_point_strategy: SigmaPointStrategy = sigma_point_strategy
        """diffbayes.utils.SigmaPointStrategy: Strategy to use for sigma point
        selection. Defaults to Julier."""

        # Sigma weights
        weights_c, weights_m = sigma_point_strategy.compute_sigma_weights()

        self.weights_c: torch.Tensor = weights_c
        """torch.Tensor: Unscented transform covariance weights. Note that this will be
        initially instantiated on the CPU, and moved in `compute_distribution()`."""
        self.weights_m: torch.Tensor = weights_m
        """torch.Tensor: Unscented transform mean weights. Note that this will be
        initially instantiated on the CPU, and moved in `compute_distribution()`."""

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
        return self.sigma_point_strategy.select_sigma_points(
            input_mean=input_mean, input_covariance=input_covariance
        )

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

        # sigma_points_centered = sigma_points - transformed_mean[:, None, :]
        sigma_points_centered = sigma_points - sigma_points[:, 0:1, :]
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
