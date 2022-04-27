import jax
import jax.numpy as np
import numpy as onp
import optax

from abc import ABC
from dataclasses import replace
from flax.training import train_state
from typing import Any

from cellori.applications.cyto.data import generate_dataset
from cellori.applications.cyto.model import CelloriCytoModel
from cellori.utils.losses import binary_cross_entropy_loss, mean_squared_error


def compute_metrics(poly_features, batch):
    vertical_gradients_pred, horizontal_gradients_pred, semantic_pred = poly_features
    mse1 = mean_squared_error(vertical_gradients_pred, 5 * batch['gradients'][:, :, :, 0:1])
    mse2 = mean_squared_error(horizontal_gradients_pred, 5 * batch['gradients'][:, :, :, 1:2])
    cel = binary_cross_entropy_loss(semantic_pred, batch['semantic'])
    loss = mse1 + mse2 + cel
    metrics = {
        'mse1': mse1,
        'mse2': mse2,
        'cel': cel,
        'total': loss
    }
    return metrics


class TrainState(train_state.TrainState, ABC):
    batch_stats: Any
    rng: Any


def create_train_state(rng, learning_rate, variables=None):
    """Creates initial `TrainState`."""
    rng, subrng = jax.random.split(rng)
    model = CelloriCytoModel()
    if variables is None:
        variables = model.init(subrng, np.ones((1, 256, 256, 2)))
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        rng=rng,
        tx=tx,
    )


def loss_fn(params, batch_stats, batch):
    poly_features, mutated_vars = CelloriCytoModel().apply(
        {'params': params, 'batch_stats': batch_stats}, batch['image'],
        mutable=['batch_stats']
    )
    metrics = compute_metrics(poly_features, batch)
    loss = metrics['total']
    return loss, (metrics, mutated_vars)


@jax.jit
def train_step(state, batch):

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (metrics, mutated_vars)), grads = grad_fn(state.params, state.batch_stats, batch)
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])

    return state, metrics


def train_epoch(state, train, batch_size, epoch):
    """Train for a single epoch."""

    rng, *subrngs = jax.random.split(state.rng, 3)
    state = replace(state, rng=rng)
    train_ds = generate_dataset(*train, subrngs[0])

    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(subrngs[1], train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: onp.mean([metrics[k] for metrics in batch_metrics_np]).astype(float)
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, mse1: %.4f, mse2: %.4f, cel: %.4f' % (
        epoch, epoch_metrics_np['total'], epoch_metrics_np['mse1'], epoch_metrics_np['mse2'], epoch_metrics_np['cel']))

    return state, epoch_metrics_np
