from typing import Any, Dict, Tuple, Union

# We don't have actual type hints for numpy, and we can't assign aliases for np.ndarray
NumpyArray = Any
# A dictionary that maps human-readable keys to numpy arrays
NumpyDict = Dict[str, NumpyArray]
# Array or dictionary of arrays
NumpyArrayOrDict = Union[NumpyDict, NumpyArray]

# States need to be raw arrays
StatesNumpy = NumpyArray
# Observations can be either numpy arrays or str->array dicts
ObservationsNumpy = NumpyArrayOrDict
# Controls can be either numpy arrays or str->array dicts
ControlsNumpy = NumpyArrayOrDict

# Trajectories are defined as (state, obs, control) tuples
TrajectoryTupleNumpy = Tuple[StatesNumpy, ObservationsNumpy, ControlsNumpy]
