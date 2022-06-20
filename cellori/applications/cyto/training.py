import jax
import jax.numpy as np
import numpy as onp
import optax

from abc import ABC
from dataclasses import replace
from flax.training import train_state
from tqdm.auto import tqdm
from typing import Any

from cellori.applications.cyto.model import CelloriCytoModel
from cellori.applications.cyto.data import transform_dataset
from cellori.utils.losses import binary_cross_entropy_loss, mean_squared_error


def compute_metrics(poly_features, batch):
    gradients_pred, semantic_pred = poly_features
    mse1 = mean_squared_error(gradients_pred[:, :, :, 0], 5 * batch['gradients'][:, :, :, 0])
    mse2 = mean_squared_error(gradients_pred[:, :, :, 1], 5 * batch['gradients'][:, :, :, 1])
    cel = binary_cross_entropy_loss(semantic_pred, batch['semantic'])
    loss = mse1 + mse2 + cel
    metrics = {
        'mse1': mse1,
        'mse2': mse2,
        'cel': cel,
        'loss': loss
    }
    return metrics


class TrainState(train_state.TrainState, ABC):
    batch_stats: Any
    rng: Any


def create_train_state(rng, learning_rate, variables=None):
    """Creates initial `TrainState`."""
    rng, *subrngs = jax.random.split(rng, 3)
    model = CelloriCytoModel()
    if variables is None:
        variables = model.init({'params': subrngs[0], 'dropout': subrngs[1]}, np.ones((1, 256, 256, 2)))
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        rng=rng,
        tx=tx,
    )


def loss_fn(params, batch_stats, rng, batch):

    if rng is None:
        poly_features = CelloriCytoModel().apply(
            {'params': params, 'batch_stats': batch_stats}, batch['image'],
            train=False
        )
        mutated_vars = None
    else:
        poly_features, mutated_vars = CelloriCytoModel().apply(
            {'params': params, 'batch_stats': batch_stats}, batch['image'],
            train=True, mutable=['batch_stats'], rngs={'dropout': rng}
        )
    metrics = compute_metrics(poly_features, batch)
    loss = metrics['loss']
    return loss, (metrics, mutated_vars)


@jax.jit
def train_step(state, train_batch, val_batch, rng):

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (train_metrics, mutated_vars)), grads = grad_fn(state.params, state.batch_stats, rng, train_batch)
    _, (val_metrics, _) = loss_fn(state.params, state.batch_stats, None, val_batch)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])
    metrics = train_metrics | {f'val_{k}': v for k, v in val_metrics.items()}

    return state, metrics


def train_epoch(state, train_ds, test_ds, batch_size, epoch):
    """Train for a single epoch."""

    print(f'epoch: {epoch}')

    rng, *subrngs = jax.random.split(state.rng, 5)
    state = replace(state, rng=rng)
    train_ds = transform_dataset(train_ds, subrngs[0])
    test_ds = transform_dataset(test_ds, subrngs[1])

    train_ds_size = len(train_ds['image'])
    val_ds_size = len(test_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    val_batch_size = val_ds_size // steps_per_epoch

    perms = jax.random.permutation(subrngs[2], train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    val_perms = jax.random.permutation(subrngs[3], val_ds_size)
    val_perms = val_perms[:steps_per_epoch * val_batch_size]  # skip incomplete batch
    val_perms = val_perms.reshape((steps_per_epoch, val_batch_size))

    batch_metrics = []
    for perm, val_perm in tqdm(zip(perms, val_perms), total=steps_per_epoch):
        rng, subrng = jax.random.split(rng)
        train_batch = {k: v[perm, ...] for k, v in train_ds.items()}
        val_batch = {k: v[val_perm, ...] for k, v in test_ds.items()}
        state, metrics = train_step(state, train_batch, val_batch, subrng)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: onp.mean([metrics[k] for metrics in batch_metrics]).astype(float)
        for k in batch_metrics[0]}

    summary = (
        f"(train) loss: {epoch_metrics['loss']:>6.3f}, mse1: {epoch_metrics['mse1']:>6.3f}, "
        f"mse2: {epoch_metrics['mse2']:>6.3f}, cel: {epoch_metrics['cel']:>6.3f}\n"
        f"(val)   loss: {epoch_metrics['val_loss']:>6.3f}, mse1: {epoch_metrics['val_mse1']:>6.3f}, "
        f"mse2: {epoch_metrics['val_mse2']:>6.3f}, cel: {epoch_metrics['val_cel']:>6.3f}\n"
    )
    print(summary)

    return state, epoch_metrics
