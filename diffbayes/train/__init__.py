"""Reference implementations for training state estimators with learnable parameters.

These are written with a custom
[model manager](https://brentyi.github.io/fannypack/utils/buddy.html) for brevity, but
can be easily translated to raw PyTorch.
"""

from ._train_dynamics import train_dynamics_recurrent, train_dynamics_single_step
from ._train_filter import train_filter
from ._train_particle_filter_measurement_model import (
    train_particle_filter_measurement_model,
)
