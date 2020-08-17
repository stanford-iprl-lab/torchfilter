import abc
import warnings
from typing import Optional, Tuple

import torch

from .. import types


class SigmaPointStrategy(abc.ABC):
    """Strategy to use for computing sigma weights + selecting sigma points.
    """

    def __init__(self, *, dim: int, lambd: float):
        self._dim = dim
        self._lambd = lambd

        if lambd + dim < 1e-3:
            warnings.warn(
                "Unscented transform parameters may result in a very small matrix root; "
                "consider tuning.",
                RuntimeWarning,
                stacklevel=1,
            )

    @abc.abstractmethod
    def compute_sigma_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
                float32 tensors on the CPU.
        """
        pass

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
        matrix_root = torch.cholesky(
            (dim + self._lambd) * input_covariance, upper=True,
        )
        assert matrix_root.shape == (N, dim, dim)

        sigma_point_offsets = input_mean.new_zeros((N, 2 * dim + 1, dim))
        sigma_point_offsets[:, 1 : 1 + dim] = matrix_root
        sigma_point_offsets[:, 1 + dim :] = -matrix_root

        # Create & return matrix of sigma points
        sigma_points: torch.Tensor = input_mean[:, None, :] + sigma_point_offsets
        assert sigma_points.shape == (N, 2 * dim + 1, dim)
        return sigma_points


class MerweSigmaPointStrategy(SigmaPointStrategy):
    """Sigma point selection in the style of [1].

    [1] http://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf

    Keyword Args:
        alpha (float, optional): Spread parameter. Defaults to `1e-2`.
        kappa (float, optional): Secondary scaling parameter, which is typically set to
            `0.0` or `3 - dim`. Defaults to `3 - dim`.
        beta (float, optional): Extra sigma parameter. Defaults to `2` (optimal for
            Gaussians, as per [1]).
    """

    def __init__(
        self,
        dim: int,
        *,
        alpha: float = 1e-2,
        kappa: Optional[float] = None,
        beta: float = 2.0,
    ):
        if kappa is None:
            # Default value for kappa
            kappa = 3 - dim

        self._alpha: float = alpha
        self._beta: float = beta
        self._kappa: float = kappa

        lambd: float = (alpha ** 2) * (dim + kappa) - dim
        super().__init__(dim=dim, lambd=lambd)

    def compute_sigma_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
                float32 tensors on the CPU.
        """

        # Create covariance weights
        weights_c = torch.full(
            size=(2 * self._dim + 1,),
            fill_value=1.0 / (2.0 * (self._dim + self._lambd)),
            dtype=torch.float32,
        )
        weights_c[0] = self._lambd / (self._dim + self._lambd) + (
            1.0 - self._alpha ** 2 + self._beta
        )

        # Mean weights should be identical, except for the first weight
        weights_m = weights_c.clone()
        weights_m[0] = self._lambd / (self._dim + self._lambd)

        return weights_c, weights_m


class JulierSigmaPointStrategy(SigmaPointStrategy):
    """Sigma point selection in this style of [1].

    [1] https://www.cs.unc.edu/~welch/kalman/media/pdf/Julier1997_SPIE_KF.pdf

    Keyword Args:
        lambd (float, optional): Spread parameter; sometimes denoted as kappa. Defaults
            to `3 - dim`.
    """

    def __init__(
        self, dim: int, *, lambd: Optional[float] = None,
    ):
        if lambd is None:
            # Default value for kappa
            lambd = 3 - dim

        super().__init__(dim=dim, lambd=lambd)

    def compute_sigma_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
                float32 tensors on the CPU.
        """

        # Create covariance weights
        weights_c = torch.full(
            size=(2 * self._dim + 1,),
            fill_value=1.0 / (2.0 * (self._dim + self._lambd)),
            dtype=torch.float32,
        )
        weights_c[0] = self._lambd / (self._dim + self._lambd)

        # Mean weights should be identical
        weights_m = weights_c.clone()

        return weights_c, weights_m
