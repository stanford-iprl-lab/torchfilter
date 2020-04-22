from typing import Callable

import fannypack
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import diffbayes


def train_dynamics_recurrent(
    buddy: fannypack.utils.Buddy,
    dynamics_model: diffbayes.base.DynamicsModel,
    dataloader: torch.utils.data.DataLoader,
    *,
    loss_function: Callable[
        [torch.tensor, torch.tensor], torch.tensor
    ] = F.mse_loss,
    log_interval: int = 1,
):
    # Dataloader should load a SubsequenceDataset
    assert isinstance(
        dataloader.dataset, diffbayes.datasets.SubsequenceDataset
    )

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # Move data
        batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
        true_states, observations, controls = batch_gpu

        # Sanity checks
        N, sequence_length, state_dim = states.shape
        assert state_dim == dynamics_model.state_dim
        assert fannypack.utils.SliceWrapper(observations).shape[:2] == (
            N,
            sequence_length,
        )
        assert fannypack.utils.SliceWrapper(controls).shape[:2] == (
            N,
            sequence_length,
        )

        # Forward pass from the first state
        initial_states = true_states[:, 0]
        state_predictions = dynamics_model.forward_loop(
            initial_states=initial_states, controls=controls[1:]
        )
        assert state_predictions.shape == (N, sequence_length - 1, state_dim)

        # Minimize MSE
        loss = loss_function(state_predictions, true_states[:, 1:])
        epoch_loss += fannypack.utils.to_numpy(loss)
        buddy.minimize()

        # Logging
        if batch_idx % log_interval == 0:
            with buddy.log_scope("train_dynamics_recurrent"):
                buddy.log("loss", loss)

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("Epoch training loss: ", epoch_loss)
