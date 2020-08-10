from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes
import fannypack


def train_virtual_sensor_model(
    buddy: fannypack.utils.Buddy,
    virtual_sensor_model: diffbayes.base.VirtualSensorModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
    optimizer_name="train_measurement",
) -> None:
    """Optimizes a kalman measurement model prediction accuracy.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        measurement_model (diffbayes.base.KalmanFilterMeasurementModel): Model to train.
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

    # Train measurement model for 1 epoch
    for batch in tqdm(dataloader):
        # Transfer to GPU and pull out batch data
        batch_gpu = fannypack.utils.to_device(batch, buddy.device)
        initial_states, next_states, observations, controls = batch_gpu

        virtual_observation, _virtual_observation_scale_tril = virtual_sensor_model(
            observations=observations
        )

        loss = loss_function(next_states, virtual_observation)
        epoch_loss += fannypack.utils.to_numpy(loss)

        buddy.minimize(loss, optimizer_name=optimizer_name)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("train_measurement"):
                buddy.log("Training loss", loss)

                buddy.log("Pred likelihoods mean", virtual_observation.mean())
                buddy.log("Pred likelihoods std", virtual_observation.std())

                buddy.log("Label likelihoods mean", next_states.mean())
                buddy.log("Label likelihoods std", next_states.std())

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("(train_kalman_filter_measurement_model) Epoch training loss: ", epoch_loss)
