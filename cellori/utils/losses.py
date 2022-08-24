import jax.numpy as np

from jax import vmap

from cellori.utils.spots import colocalize_pixels


def colocalization_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels):

    vmap_colocalization_loss = vmap(_colocalization_loss, in_axes=(0, 0, 0, 0, 0))
    cl_sl1, cl_bcel, cl_invf1 = vmap_colocalization_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels)

    return np.mean(cl_sl1), np.mean(cl_bcel), np.mean(cl_invf1)


def _colocalization_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels):

    labels_pred = labels_pred[:, :, 0]
    labels = labels[:, :, 0]
    dilated_labels = dilated_labels[:, :, 0]

    cl_sl1 = np.sum(smooth_l1(deltas_pred, deltas) * dilated_labels) / np.sum(dilated_labels)
    cl_bcel = binary_cross_entropy_loss(labels_pred, labels, weighted=True)

    counts = colocalize_pixels(deltas_pred * dilated_labels[:, :, None], labels_pred)

    tp = np.sum(dilated_labels * counts)
    fp = np.sum(labels_pred) - tp

    num_captured = np.sum(labels_pred * labels)
    num_uncaptured = np.sum(labels) - num_captured
    colocalization_area = tp / (num_captured + 1e-07)
    fn = num_uncaptured * colocalization_area

    precision = tp / (tp + fp + 1e-07)
    recall = tp / (tp + fn + 1e-07)
    f1 = 2 * precision * recall / (precision + recall + 1e-07)

    cl_invf1 = 1 - f1

    return cl_sl1, cl_bcel, cl_invf1


def discriminative_loss(y_pred, masks, regions, delta=4, alpha=1, beta=1, gamma=0.001):

    vmap_discriminative_loss = vmap(_discriminative_loss, in_axes=(0, 0, 0, None, None, None, None))

    return np.mean(vmap_discriminative_loss(y_pred, masks, regions, delta, alpha, beta, gamma))


def _discriminative_loss(y_pred, mask, regions, delta, alpha, beta, gamma):

    c = mask.max()
    labels = vmap((lambda x, n: x == n), in_axes=(None, 0))(mask, regions)

    def statistics(y, label):

        n = np.sum(label) + 1e-07
        x = np.where(label, y, 0)
        mu = 1 / n * np.sum(x, axis=(0, 1))

        return n, mu

    ns, mus = vmap(statistics, in_axes=(None, 0))(y_pred, labels)

    def variance(y, label, n, mu):

        return 1 / n * np.sum(np.linalg.norm(np.where(label, mu - y, 0), ord=1, axis=2) ** 2)

    dl_var = 1 / c * np.sum(vmap(variance, in_axes=(None, 0, 0, 0))(y_pred, labels, ns, mus))

    def distance(mu1, mu2):

        return np.maximum(0, 2 * delta - np.linalg.norm(mu1 - mu2, ord=1)) ** 2

    m = np.ones((len(mus), len(mus))) - np.eye(len(mus))
    dl_dist = 1 / (c * (c - 1)) * np.sum(m * vmap(vmap(distance, in_axes=(0, None)), in_axes=(None, 0))(mus, mus))

    def regularization(mu):

        return np.linalg.norm(mu, ord=1)

    dl_reg = 1 / c * np.sum(vmap(regularization, in_axes=0)(mus))

    dl = alpha * dl_var + beta * dl_dist + gamma * dl_reg

    return dl


def mean_squared_error(y_pred, y_true):
    mse = np.mean((y_true - y_pred) ** 2)

    return mse


def smooth_l1(y_pred, y_true, beta=0.1):

    diff = y_pred - y_true
    l1 = np.linalg.norm(diff, ord=1, axis=-1)
    l2 = np.linalg.norm(diff, ord=2, axis=-1)
    criteria = l1 < beta

    sl1l = 0
    sl1l = sl1l + criteria * 0.5 * l2 / beta
    sl1l = sl1l + (~criteria) * (l1 - 0.5 * beta)

    return sl1l


def binary_cross_entropy_loss(p, labels, weighted=False):
    if weighted:
        beta = np.sum(~labels) / (np.sum(labels) + 1)
    else:
        beta = 1

    bcel = -np.mean(beta * np.log(p + 1e-7) * labels + np.log((1 - p) + 1e-7) * (1 - labels))

    return bcel


def binary_focal_loss(p, labels, gamma=2, weighted=False):
    if weighted:
        beta = np.sum(~labels) / (np.sum(labels) + 1)
    else:
        beta = 1

    bfl = -np.mean(beta * (1 - p) ** gamma * np.log(p + 1e-7) * labels +
                   p ** gamma * np.log((1 - p) + 1e-7) * (1 - labels))

    return bfl


def cross_entropy_loss(p, labels, alpha=0):
    if alpha > 0:
        beta = _calculate_class_weights(labels, alpha).reshape(1, 1, 1, -1)
    else:
        beta = 1

    cel = -np.mean(beta * np.log(p + 1e-7) * labels)

    return cel


def focal_loss(p, labels, gamma=2, alpha=0):
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
