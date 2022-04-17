import jax.numpy as np


def mean_squared_error(y_pred, y_true):
    mse = np.mean((y_true - y_pred) ** 2)

    return mse


def binary_cross_entropy_loss(p, labels, weighted=False):
    if weighted:
        alpha = 2 * (1 - np.sum(labels) / np.prod(np.array(labels.shape)))
    else:
        alpha = 1

    cel = -np.mean(alpha * np.log(p + 1e-7) * labels + (2 - alpha) * np.log((1 - p) + 1e-7) * (1 - labels))

    return cel


def cross_entropy_loss(p, labels, weighted=False):
    if weighted:
        alpha = _calculate_class_weights(labels).reshape(1, 1, 1, -1)
    else:
        alpha = 1

    cel = -np.mean(alpha * np.log(p + 1e-7) * labels)

    return cel


def focal_loss(p, labels, gamma, weighted=False):
    if weighted:
        alpha = _calculate_class_weights(labels).reshape(1, 1, -1)
    else:
        alpha = 1

    fl = -np.mean(alpha * (1 - p) ** gamma * np.log(p + 1e-7) * labels)

    return fl


def _calculate_class_weights(labels):
    num_classes = labels.shape[-1]
    class_sums = np.sum(labels, axis=(0, 1, 2))
    total_sum = np.sum(class_sums)
    class_weights = 1 / num_classes * (total_sum / (class_sums + 1))

    return class_weights
