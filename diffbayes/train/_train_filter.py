from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes
import fannypack


def train_filter(
    buddy: fannypack.utils.Buddy,
    filter_model: diffbayes.base.Filter,
    dataloader: DataLoader,
    initial_covariance: torch.Tensor,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 1,
) -> None:
    """Trains a filter end-to-end via backpropagation through time.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        filter_model (diffbayes.base.DynamicsModel): Model to train.
        dataloader (DataLoader): Loader for a SubsequenceDataset.
        initial_covariance (torch.Tensor): Covariance matrix of error in initial
            posterior, whose mean is sampled from a Gaussian centered at the
            ground-truth start state. Shape should be (state_dim, state_dim).

    Keyword Args:
        loss_function (callable, optional): Loss function, from `torch.nn.functional`.
            Defaults to MSE.
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Dataloader should load a SubsequenceDataset
    assert isinstance(dataloader.dataset, diffbayes.datasets.SubsequenceDataset)
    assert initial_covariance.shape == (filter_model.state_dim, filter_model.state_dim)

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train filter model for 1 epoch
    with buddy.log_scope("train_filter_recurrent"):
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            # Move data
            batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
            true_states, observations, controls = batch_gpu

            # Sanity checks
            N, sequence_length, state_dim = true_states.shape
            assert state_dim == filter_model.state_dim
            assert fannypack.utils.SliceWrapper(observations).shape[:2] == (
                N,
                sequence_length,
            )
            assert fannypack.utils.SliceWrapper(controls).shape[:2] == (
                N,
                sequence_length,
            )

            # Populate initial filter belief
            initial_states_covariance = initial_covariance[None, :, :].expand(
                (N, state_dim, state_dim)
            )
            initial_states = torch.distributions.MultivariateNormal(
                true_states[:, 0], initial_states_covariance
            ).sample()
            filter_model.initialize_beliefs(
                mean=initial_states, covariance=initial_states_covariance
            )

            # Forward pass from the first state
            state_predictions = filter_model.forward_loop(
                observations=observations[1:], controls=controls[1:]
            )
            assert state_predictions.shape == (N, sequence_length - 1, state_dim)

            # Minimize loss
            loss = loss_function(state_predictions, true_states[:, 1:])
            buddy.minimize(loss, optimizer_name="train_filter_recurrent")
            epoch_loss += fannypack.utils.to_numpy(loss)

            # Logging
            if batch_idx % log_interval == 0:
                buddy.log("loss", loss)

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("Epoch training loss: ", epoch_loss)
