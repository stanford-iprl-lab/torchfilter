from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes
import fannypack


def train_virtual_sensor(
    buddy: fannypack.utils.Buddy,
    virtual_sensor_model: diffbayes.base.VirtualSensorModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
    optimizer_name="train_measurement",
) -> None:
    """Optimizes a virtual sensor model's prediction accuracy. Minimizes output mean
    error only; does not define a loss on uncertainty.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        measurement_model (diffbayes.base.VirtualSensorModel): Model to train.
        dataloader (DataLoader): Loader for a SingleStepDataset.

    Keyword Args:
        loss_function (callable, optional): Loss function, from `torch.nn.functional`.
            Defaults to MSE.
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Dataloader should load a SingleStepDataset
    assert isinstance(dataloader.dataset, diffbayes.data.SingleStepDataset)
    assert virtual_sensor_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train virtual sensor model for 1 epoch
    for batch in tqdm(dataloader):
        # Transfer to GPU and pull out batch data
        batch_gpu = fannypack.utils.to_device(batch, buddy.device)
        previous_states, states, observations, controls = batch_gpu

        virtual_observation, _virtual_observation_scale_tril = virtual_sensor_model(
            observations=observations
        )

        loss = loss_function(states, virtual_observation)
        epoch_loss += fannypack.utils.to_numpy(loss)

        buddy.minimize(loss, optimizer_name=optimizer_name)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("train_virtual_sensor"):
                buddy.log_scalar("Training loss", loss)

                buddy.log_scalar("Pred likelihoods mean", virtual_observation.mean())
                buddy.log_scalar("Pred likelihoods std", virtual_observation.std())

                buddy.log_scalar("Label likelihoods mean", states.mean())
                buddy.log_scalar("Label likelihoods std", states.std())

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("(train_virtual_sensor_model) Epoch training loss: ", epoch_loss)
