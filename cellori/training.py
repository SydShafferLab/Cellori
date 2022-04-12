import jax
import jax.numpy as np
import numpy as onp
import optax

from abc import ABC
from flax.training import train_state
from typing import Any

from .cellori import Cellori
from .losses import focal_loss, mean_squared_error


def compute_metrics(poly_features, batch):
    distance_transform_pred, class_transform_pred = poly_features
    mse = mean_squared_error(distance_transform_pred, batch['distance_transform'])
    fl = focal_loss(class_transform_pred, batch['class_transform'], gamma=2, weighted=True)
    loss = mse + fl
    metrics = {
        'mse': mse,
        'fl': fl,
        'total': loss
    }
    return metrics


class TrainState(train_state.TrainState, ABC):
    batch_stats: Any


def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    model = Cellori()
    variables = model.init(rng, np.ones((1, 384, 384, 2)))
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=tx,
    )


def loss_fn(params, batch_stats, batch):
    poly_features, mutated_vars = Cellori().apply(
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


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
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
        k: onp.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, mse: %.4f, fl: %.4f' % (
        epoch, epoch_metrics_np['total'], epoch_metrics_np['mse'], epoch_metrics_np['fl']))

    return state
