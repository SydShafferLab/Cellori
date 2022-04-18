import jax.numpy as np


def mean_squared_error(y_pred, y_true):
    mse = np.mean((y_true - y_pred) ** 2)

    return mse


def binary_cross_entropy_loss(p, labels, alpha=0):
    if alpha > 0:
        class_sums = np.array([np.sum(labels), np.sum(~labels)])
        class_sums = (1 / (class_sums + 1)) ** alpha
        beta = 2 * (1 - class_sums[0] / class_sums.sum())
    else:
        beta = 1

    cel = -np.mean(beta * np.log(p + 1e-7) * labels + (2 - beta) * np.log((1 - p) + 1e-7) * (1 - labels))

    return cel


def cross_entropy_loss(p, labels, alpha=0):
    if alpha > 0:
        beta = _calculate_class_weights(labels, alpha).reshape(1, 1, 1, -1)
    else:
        beta = 1

    cel = -np.mean(beta * np.log(p + 1e-7) * labels)

    return cel


def focal_loss(p, labels, gamma, alpha=0):
    if alpha > 0:
        beta = _calculate_class_weights(labels, alpha).reshape(1, 1, -1)
    else:
        beta = 1

    fl = -np.mean(beta * (1 - p) ** gamma * np.log(p + 1e-7) * labels)

    return fl


def _calculate_class_weights(labels, alpha):

    class_sums = np.sum(labels, axis=(0, 1, 2))
    class_sums = (1 / (class_sums + 1)) ** alpha
    beta = class_sums / class_sums.sum()

    return beta
