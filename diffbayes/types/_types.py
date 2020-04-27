from typing import Any, Dict, Tuple, Union

# Make an explicit list of names to expose
__all__ = [
    "NumpyArray",
    "NumpyDict",
    "NumpyArrayOrDict",
    "StatesNumpy",
    "ObservationsNumpy",
    "ControlsNumpy",
    "TrajectoryTupleNumpy",
]

NumpyArray = Any
""" Generic type hint for `np.ndarray` objects. """

NumpyDict = Dict[str, NumpyArray]
""" Dictionary from `str` keys to `np.ndarray` values. """

NumpyArrayOrDict = Union[NumpyArray, NumpyDict]
""" Union of NumpyArray and NumpyDict types. """

StatesNumpy = NumpyArray
""" State array type hint. Needs to be a raw `np.ndarray`. """

ObservationsNumpy = NumpyArrayOrDict
""" Observations can be either `np.ndarray` objects or `str->nd.ndarray` dictionaries. """

ControlsNumpy = NumpyArrayOrDict
""" Observations can be either `np.ndarray` objects or `str->np.ndarray` dictionaries. """

TrajectoryTupleNumpy = Tuple[StatesNumpy, ObservationsNumpy, ControlsNumpy]
""" Trajectories are defined as `(States, Observation, Controls)` tuples. """
