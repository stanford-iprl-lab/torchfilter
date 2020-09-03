from typing import Callable

import fannypack
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffbayes


def train_kalman_filter_measurement(
    buddy: fannypack.utils.Buddy,
    measurement_model: diffbayes.base.KalmanFilterMeasurementModel,
    dataloader: DataLoader,
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    log_interval: int = 10,
    optimizer_name="train_kalman_filter_measurement",
) -> None:
    """Optimizes a Kalman filter measurement model's prediction accuracy. Minimizes
    output mean error only; does not define a loss on uncertainty.

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
    assert measurement_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train measurement model for 1 epoch
    for batch in tqdm(dataloader):
        # Transfer to GPU and pull out batch data
        batch_gpu = fannypack.utils.to_device(batch, buddy.device)
        previous_states, states, observations, controls = batch_gpu

        predicted_observations, _predicted_observations_scale_tril = measurement_model(
            states=states
        )

        loss = loss_function(observations, predicted_observations)
        epoch_loss += fannypack.utils.to_numpy(loss)

        buddy.minimize(loss, optimizer_name=optimizer_name)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("train_measurement"):
                buddy.log_scalar("Training loss", loss)

                buddy.log_scalar("Pred likelihoods mean", predicted_observations.mean())
                buddy.log_scalar("Pred likelihoods std", predicted_observations.std())

                buddy.log_scalar("Label likelihoods mean", states.mean())
                buddy.log_scalar("Label likelihoods std", states.std())

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("(train_kalman_filter_measurement_model) Epoch training loss: ", epoch_loss)
