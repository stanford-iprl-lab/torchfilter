from typing import Any, Dict, Tuple, Union

import torch

# Make an explicit list of names to expose
__all__ = [
    "NumpyArray",
    "TorchTensor",
    "NumpyDict",
    "TorchDict",
    "NumpyArrayOrDict",
    "TorchTensorOrDict",
    "StatesNumpy",
    "StatesTorch",
    "ObservationsNumpy",
    "ObservationsTorch",
    "ControlsNumpy",
    "ControlsTorch",
    "TrajectoryTupleNumpy",
    "TrajectoryTupleTorch",
]

NumpyArray = Any
""" Type hint for `np.ndarray` objects. (alias for `typing.Any`) """
TorchTensor = torch.Tensor
""" Type hint for `torch.Tensor` objects. (alias for `torch.Tensor`) """

NumpyDict = Dict[str, NumpyArray]
""" Dictionary from `str` keys to `np.ndarray` values. """
TorchDict = Dict[str, TorchTensor]
""" Dictionary from `str` keys to `torch.Tensor` values. """

NumpyArrayOrDict = Union[NumpyArray, NumpyDict]
""" Union of NumpyArray and NumpyDict types. """
TorchTensorOrDict = Union[TorchTensor, TorchDict]
""" Union of TorchTensor and TorchDict types. """

StatesNumpy = NumpyArray
""" State array type hint. Needs to be a raw `np.ndarray`. """
StatesTorch = TorchTensor
""" State array type hint. Needs to be a raw `torch.Tensor`. """

ObservationsNumpy = NumpyArrayOrDict
""" Observations can be either `np.ndarray` objects or `str->np.ndarray` dictionaries. """
ObservationsTorch = TorchTensorOrDict
""" Observations can be either `torch.Tensor` objects or `str->torch.Tensor` dictionaries. """

ControlsNumpy = NumpyArrayOrDict
""" Controls can be either `np.ndarray` objects or `str->np.ndarray` dictionaries. """
ControlsTorch = TorchTensorOrDict
""" Controls can be either `torch.Tensor` objects or `str->torch.Tensor` dictionaries. """

TrajectoryTupleNumpy = Tuple[StatesNumpy, ObservationsNumpy, ControlsNumpy]
""" Trajectories in Numpy, defined as `(States, Observation, Controls)` tuples. """
TrajectoryTupleTorch = Tuple[StatesTorch, ObservationsTorch, ControlsTorch]
""" Trajectory in PyTorch, defined as `(States, Observation, Controls)` tuples. """
