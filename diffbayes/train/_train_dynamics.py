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
    loss_function: str = "nll",
    log_interval: int = 10,
) -> None:
    """Optimizes a dynamics model's single-step prediction accuracy.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        dynamics_model (diffbayes.base.DynamicsModel): Model to train.
        dataloader (DataLoader): Loader for a SingleStepDataset.

    Keyword Args:
        loss_function (str, optional): Either "nll" for negative log-likelihood or "mse"
            for mean-squared error. Defaults to "nll".
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Input validation
    assert isinstance(dataloader.dataset, diffbayes.data.SingleStepDataset)
    assert loss_function in ("nll", "mse")
    assert dynamics_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
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
        predictions, scale_trils = dynamics_model(
            initial_states=initial_states, controls=controls
        )
        assert predictions.shape == (N, state_dim)

        # Check if we want to log
        log_flag = batch_idx % log_interval == 0

        # Minimize loss
        losses = {}
        if log_flag or loss_function == "mse":
            losses["mse"] = F.mse_loss(predictions, next_states)
        if log_flag or loss_function == "nll":
            log_likelihoods = torch.distributions.MultivariateNormal(
                loc=predictions, scale_tril=scale_trils
            ).log_prob(next_states)
            assert log_likelihoods.shape == (N,)
            losses["nll"] = -torch.sum(log_likelihoods)

        buddy.minimize(
            losses[loss_function], optimizer_name="train_dynamics_single_step"
        )
        epoch_loss += fannypack.utils.to_numpy(losses[loss_function])

        # Logging
        if log_flag:
            with buddy.log_scope("train_dynamics_single_step"):
                buddy.log_scalar("MSE loss", losses["mse"])
                buddy.log_scalar("NLL loss", losses["nll"])

    # Print average training loss
    epoch_loss /= len(dataloader)
    print(
        f"(train_dynamics_single_step) Epoch training loss ({loss_function}): {epoch_loss}"
    )


def train_dynamics_recurrent(
    buddy: fannypack.utils.Buddy,
    dynamics_model: diffbayes.base.DynamicsModel,
    dataloader: DataLoader,
    *,
    loss_function: str = "nll",
    log_interval: int = 10,
) -> None:
    """Trains a dynamics model via backpropagation through time.

    Args:
        buddy (fannypack.utils.Buddy): Training helper.
        dynamics_model (diffbayes.base.DynamicsModel): Model to train.
        dataloader (DataLoader): Loader for a SubsequenceDataset.

    Keyword Args:
        loss_function (str, optional): Either "nll" for negative log-likelihood or "mse"
            for mean-squared error. Defaults to "nll".
        log_interval (int, optional): Minibatches between each Tensorboard log.
    """
    # Input validation
    assert isinstance(dataloader.dataset, diffbayes.data.SubsequenceDataset)
    assert loss_function in ("nll", "mse")
    assert dynamics_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train dynamics model for 1 epoch
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        # Move data
        batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
        true_states, observations, controls = batch_gpu

        # Swap batch size, sequence length axes
        true_states = _swap_batch_sequence_axes(true_states)
        observations = fannypack.utils.SliceWrapper(observations).map(
            _swap_batch_sequence_axes
        )
        controls = fannypack.utils.SliceWrapper(controls).map(_swap_batch_sequence_axes)

        # Shape checks
        T, N, state_dim = true_states.shape
        assert state_dim == dynamics_model.state_dim
        assert fannypack.utils.SliceWrapper(observations).shape[:2] == (T, N)
        assert fannypack.utils.SliceWrapper(controls).shape[:2] == (T, N)
        assert batch_idx != 0 or N == dataloader.batch_size

        # Forward pass from the first state
        initial_states = true_states[0]
        predictions, scale_trils = dynamics_model.forward_loop(
            initial_states=initial_states, controls=controls[1:]
        )
        assert predictions.shape == (T - 1, N, state_dim)

        # Check if we want to log
        log_flag = batch_idx % log_interval == 0

        # Minimize loss
        losses = {}
        if log_flag or loss_function == "mse":
            losses["mse"] = F.mse_loss(predictions, true_states[1:])
        if log_flag or loss_function == "nll":
            log_likelihoods = torch.distributions.MultivariateNormal(
                loc=predictions, scale_tril=scale_trils
            ).log_prob(true_states[1:])
            assert log_likelihoods.shape == (T - 1, N)
            losses["nll"] = -torch.sum(log_likelihoods)

        buddy.minimize(losses[loss_function], optimizer_name="train_dynamics_recurrent")
        epoch_loss += fannypack.utils.to_numpy(losses[loss_function])

        # Logging
        if log_flag:
            with buddy.log_scope("train_dynamics_recurrent"):
                buddy.log("MSE loss", losses["mse"])
                buddy.log("NLL loss", losses["nll"])

    # Print average training loss
    epoch_loss /= len(dataloader)
    print(
        f"(train_dynamics_recurrent) Epoch training loss ({loss_function}): {epoch_loss}"
    )
