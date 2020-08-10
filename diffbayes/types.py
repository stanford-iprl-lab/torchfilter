"""Semantic typehints for filtering.
"""
import dataclasses
from typing import Dict, Union

import numpy as np
import torch

# Make an explicit list of names to expose
__all__ = [
    "NumpyDict",
    "TorchDict",
    "NumpyArrayOrDict",
    "TorchTensorOrDict",
    "StatesNumpy",
    "StatesTorch",
    "ObservationsNumpy",
    "ObservationsTorch",
    "ObservationsNoDictNumpy",
    "ObservationsNoDictTorch",
    "ControlsNumpy",
    "ControlsTorch",
    "CovarianceTorch",
    "ScaleTrilTorch",
    "TrajectoryNumpy",
]

NumpyDict = Dict[str, np.ndarray]
"""Dictionary from `str` keys to `np.ndarray` values."""
TorchDict = Dict[str, torch.Tensor]
"""Dictionary from `str` keys to `torch.Tensor` values."""

NumpyArrayOrDict = Union[np.ndarray, NumpyDict]
"""Union of np.ndarray and NumpyDict types."""
TorchTensorOrDict = Union[torch.Tensor, TorchDict]
"""Union of torch.Tensor and TorchDict types."""

StatesNumpy = np.ndarray
"""State array type hint. Needs to be a raw `np.ndarray`."""
StatesTorch = torch.Tensor
"""State array type hint. Needs to be a raw `torch.Tensor`."""

ObservationsNumpy = NumpyArrayOrDict
"""Observations can be either `np.ndarray` objects or `str->np.ndarray` dictionaries."""
ObservationsTorch = TorchTensorOrDict
"""Observations can be either `torch.Tensor` objects or `str->torch.Tensor` dictionaries."""

ObservationsNoDictNumpy = np.ndarray
"""Same as `ObservationsNumpy`, but no dictionaries."""
ObservationsNoDictTorch = torch.Tensor
"""Same as `ObservationsTorch`, but no dictionaries."""

ControlsNumpy = NumpyArrayOrDict
"""Controls can be either `np.ndarray` objects or `str->np.ndarray` dictionaries."""
ControlsTorch = TorchTensorOrDict
"""Controls can be either `torch.Tensor` objects or `str->torch.Tensor` dictionaries."""

CovarianceTorch = torch.Tensor
"""Covariance matrix as `torch.Tensor`. Must be positive semi-definite."""
ScaleTrilTorch = torch.Tensor
"""Lower-triangular cholesky decomposition of covariance matrix as `torch.Tensor`."""


@dataclasses.dataclass
class TrajectoryNumpy:
    """A single trajectory, represented with NumPy arrays.
    """

    states: StatesNumpy
    """np.ndarray: State array. Shape should be `(T, state_dim)`.
    """

    observations: ObservationsNumpy
    """Union[np.ndarray, Dict[str, np.ndarray]]: Observations. Shape(s) should be
    `(T, ...)`."""

    controls: ControlsNumpy
    """Union[np.ndarray, Dict[str, np.ndarray]]: Controls. Shape(s) should be
    `(T, ...)`."""
