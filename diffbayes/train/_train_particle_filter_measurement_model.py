from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes
import fannypack


def train_particle_filter_measurement_model(
    buddy: fannypack.utils.Buddy,
    measurement_model: diffbayes.base.ParticleFilterMeasurementModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
) -> None:
    """Reference implementation for pre-training a particle filter measurement model.
    Minimizes prediction error for log-likelihood outputs from (state, observation)
    pairs.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        measurement_model (diffbayes.base.ParticleFilterMeasurementModel): Model.
        dataloader (DataLoader): Loader for a ParticleFilterMeasurementDataset.

    Keyword Args:
        loss_function (callable, optional): Loss function, from `torch.nn.functional`.
            Defaults to MSE.
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Dataloader should load a ParticleFilterMeasurementDataset
    assert isinstance(
        dataloader.dataset, diffbayes.data.ParticleFilterMeasurementDataset
    )
    assert measurement_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
    with buddy.log_scope("train_measurement"):
        # Train measurement model only for 1 epoch
        for batch in tqdm(dataloader):
            # Transfer to GPU and pull out batch data
            batch_gpu = fannypack.utils.to_device(batch, buddy.device)
            noisy_states, observations, log_likelihoods = batch_gpu

            noisy_states = noisy_states[:, np.newaxis, :]
            pred_likelihoods = measurement_model(
                states=noisy_states, observations=observations
            )
            assert len(pred_likelihoods.shape) == 2
            pred_likelihoods = pred_likelihoods.squeeze(dim=1)
            assert pred_likelihoods.shape == log_likelihoods.shape

            loss = loss_function(pred_likelihoods, log_likelihoods)
            epoch_loss += fannypack.utils.to_numpy(loss)

            buddy.minimize(loss, optimizer_name="train_measurement")

            if buddy.optimizer_steps % log_interval == 0:
                buddy.log("Training loss", loss)

                buddy.log("Pred likelihoods mean", pred_likelihoods.mean())
                buddy.log("Pred likelihoods std", pred_likelihoods.std())

                buddy.log("Label likelihoods mean", log_likelihoods.mean())
                buddy.log("Label likelihoods std", log_likelihoods.std())

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("(train_particle_filter_measurement_model) Epoch training loss: ", epoch_loss)
