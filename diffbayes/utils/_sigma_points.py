"""Private module; avoid importing from directly.
"""

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from overrides import overrides


class SigmaPointStrategy(abc.ABC):
    """Strategy to use for computing sigma weights + selecting sigma points.
    """

    @abc.abstractmethod
    def compute_lambda(self, dim: int) -> float:
        """Compute sigma point scaling parameter.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            float: Lambda scaling parameter.
        """
        pass

    @abc.abstractmethod
    def compute_sigma_weights(self, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
            float32 tensors on the CPU.
        """
        pass


@dataclass(frozen=True)
class MerweSigmaPointStrategy(SigmaPointStrategy):
    """Sigma point selection in the style of [2].

    [2] http://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf

    Keyword Args:
        alpha (float): Spread parameter. Defaults to `1e-2`.
        kappa (Optional[float]): Secondary scaling parameter, which is typically set to
            `0.0` or `3 - dim`. If None, we use `3 - dim`.
        beta (float): Extra sigma parameter. Defaults to `2` (optimal for Gaussians, as
            per [1]).
    """

    alpha: float = 1e-2
    beta: float = 2.0
    kappa: Optional[float] = None

    @overrides
    def compute_lambda(self, dim: int) -> float:
        """Compute sigma point scaling parameter.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            float: Lambda scaling parameter.
        """
        kappa = 3.0 - dim if self.kappa is None else self.kappa
        return (self.alpha ** 2) * (dim + kappa) - dim

    @overrides
    def compute_sigma_weights(self, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
            float32 tensors on the CPU.
        """

        lambd = self.compute_lambda(dim=dim)

        # Create covariance weights
        weights_c = torch.full(
            size=(2 * dim + 1,),
            fill_value=1.0 / (2.0 * (dim + lambd)),
            dtype=torch.float32,
        )
        weights_c[0] = lambd / (dim + lambd) + (1.0 - self.alpha ** 2 + self.beta)

        # Mean weights should be identical, except for the first weight
        weights_m = weights_c.clone()
        weights_m[0] = lambd / (dim + lambd)

        return weights_c, weights_m


@dataclass(frozen=True)
class JulierSigmaPointStrategy(SigmaPointStrategy):
    """Sigma point selection in this style of [1].

    [1] https://www.cs.unc.edu/~welch/kalman/media/pdf/Julier1997_SPIE_KF.pdf

    Keyword Args:
        lambd (Optional[float]): Spread parameter; sometimes denoted as kappa. If
            `None`, we use `3 - dim`.
    """

    lambd: Optional[float] = None

    @overrides
    def compute_lambda(self, dim: int) -> float:
        """Compute sigma point scaling parameter.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            float: Lambda scaling parameter.
        """
        return 3.0 - dim if self.lambd is None else self.lambd

    @overrides
    def compute_sigma_weights(self, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for computing sigma weights.

        Args:
            dim (int): Dimensionality of input vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
            float32 tensors on the CPU.
        """

        lambd = self.compute_lambda(dim=dim)

        # Create covariance weights
        weights_c = torch.full(
            size=(2 * dim + 1,),
            fill_value=1.0 / (2.0 * (dim + lambd)),
            dtype=torch.float32,
        )
        weights_c[0] = lambd / (dim + lambd)

        # Mean weights should be identical
        weights_m = weights_c.clone()

        return weights_c, weights_m
