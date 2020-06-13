from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes
import fannypack


def _swap_batch_sequence_axes(tensor):
    """Converts data formatted as (N, T, ...) to (T, N, ...)
    """
    return torch.transpose(tensor, 0, 1)


def train_dynamics_single_step(
    buddy: fannypack.utils.Buddy,
    dynamics_model: diffbayes.base.DynamicsModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
) -> None:
    """Optimizes a dynamics model's single-step prediction accuracy.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        dynamics_model (diffbayes.base.DynamicsModel): Model to train.
        dataloader (DataLoader): Loader for a SingleStepDataset.

    Keyword Args:
        loss_function (callable, optional): Loss function, from `torch.nn.functional`.
            Defaults to MSE.
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Dataloader should load a SingleStepDataset
    assert isinstance(dataloader.dataset, diffbayes.data.SingleStepDataset)
    assert dynamics_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
    with buddy.log_scope("train_dynamics_single_step"):
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            # Move data
            batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
            initial_states, next_states, observations, controls = batch_gpu

            # Sanity checks
            N, state_dim = initial_states.shape
            assert next_states.shape == initial_states.shape
            assert state_dim == dynamics_model.state_dim
            assert fannypack.utils.SliceWrapper(observations).shape[:1] == (N,)
            assert fannypack.utils.SliceWrapper(controls).shape[:1] == (N,)

            # Single-step prediction
            state_predictions = dynamics_model(
                initial_states=initial_states, controls=controls, noisy=False
            )
            assert state_predictions.shape == (N, state_dim)

            # Minimize loss
            loss = loss_function(state_predictions, next_states)
            buddy.minimize(loss, optimizer_name="train_dynamics_single_step")
            epoch_loss += fannypack.utils.to_numpy(loss)

            # Logging
            if batch_idx % log_interval == 0:
                buddy.log("loss", loss)

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("Epoch training loss: ", epoch_loss)


def train_dynamics_recurrent(
    buddy: fannypack.utils.Buddy,
    dynamics_model: diffbayes.base.DynamicsModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
) -> None:
    """Trains a dynamics model via backpropagation through time.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        dynamics_model (diffbayes.base.DynamicsModel): Model to train.
        dataloader (DataLoader): Loader for a SubsequenceDataset.

    Keyword Args:
        loss_function (callable, optional): Loss function, from `torch.nn.functional`.
            Defaults to MSE.
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Dataloader should load a SubsequenceDataset
    assert isinstance(dataloader.dataset, diffbayes.data.SubsequenceDataset)

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
    with buddy.log_scope("train_dynamics_recurrent"):
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            # Move data
            batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
            true_states, observations, controls = batch_gpu

            # Swap batch size, sequence length axes
            true_states = _swap_batch_sequence_axes(true_states)
            observations = fannypack.utils.SliceWrapper(observations).map(
                _swap_batch_sequence_axes
            )
            controls = fannypack.utils.SliceWrapper(controls).map(
                _swap_batch_sequence_axes
            )

            # Shape checks
            T, N, state_dim = true_states.shape
            assert state_dim == dynamics_model.state_dim
            assert fannypack.utils.SliceWrapper(observations).shape[:2] == (T, N)
            assert fannypack.utils.SliceWrapper(controls).shape[:2] == (T, N)
            assert batch_idx != 0 or N == dataloader.batch_size

            # Forward pass from the first state
            initial_states = true_states[0]
            state_predictions = dynamics_model.forward_loop(
                initial_states=initial_states, controls=controls[1:], noisy=False
            )
            assert state_predictions.shape == (T - 1, N, state_dim)

            # Minimize loss
            loss = loss_function(state_predictions, true_states[1:])
            buddy.minimize(loss, optimizer_name="train_dynamics_recurrent")
            epoch_loss += fannypack.utils.to_numpy(loss)

            # Logging
            if batch_idx % log_interval == 0:
                buddy.log("loss", loss)

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("Epoch training loss: ", epoch_loss)
