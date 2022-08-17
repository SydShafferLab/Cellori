import jax.numpy as np
import numpy as onp

from jax import jit, vmap
from skimage import feature


def compute_spot_coordinates(deltas, labels, min_distance=1, threshold=1.5):

    counts = jit(colocalize_pixels)(deltas * labels)
    adjusted_counts = onp.asarray(counts + labels[:, :, 0] - 1)
    peaks = feature.peak_local_max(adjusted_counts,
                                   min_distance=min_distance, threshold_abs=threshold, exclude_border=False)
    num_peaks = len(peaks)
    if num_peaks > 0:
        coords = peaks + onp.asarray(deltas)[peaks[:, 0], peaks[:, 1]]
    else:
        coords = onp.empty((0, 2), dtype=onp.float32)

    return coords, adjusted_counts


def colocalize_pixels(deltas):

    i, j = np.arange(deltas.shape[0]), np.arange(deltas.shape[1])
    ii, jj = np.meshgrid(i, j, indexing='ij')
    index_map = np.stack((ii, jj), axis=-1)

    convergence = deltas + index_map

    vmap_count_convergence = vmap(vmap(_count_convergence, in_axes=(None, None, 0)), in_axes=(None, 0, None))
    counts = vmap_count_convergence(convergence, i, j)

    return counts


def _count_convergence(convergence, i, j):

    sources = _search_convergence(convergence, i, j)
    count = np.sum(sources)

    return count


def _search_convergence(convergence, i, j):

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) \
              & (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources
