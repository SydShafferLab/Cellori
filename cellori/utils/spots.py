import jax.numpy as np
import numpy as onp

from jax import vmap
from jax.lax import dynamic_slice
from skimage import feature, measure


def compute_spot_coordinates(deltas, counts, threshold, min_distance):

    stack = counts.ndim == 3
    counts = onp.asarray(counts)

    if stack:
        labels = measure.label(counts > threshold)
        peaks = onp.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)
    else:
        peaks = feature.peak_local_max(counts, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    if len(peaks) > 0:
        if stack:
            coords = peaks + onp.pad(onp.asarray(deltas)[peaks[:, 0], peaks[:, 1], peaks[:, 2]], ((0, 0), (1, 0)))
        else:
            coords = peaks + onp.asarray(deltas)[peaks[:, 0], peaks[:, 1]]
    else:
        coords = onp.empty((0, counts.ndim), dtype=onp.float32)

    return coords


def colocalize_pixels(deltas, labels):

    i, j = np.arange(deltas.shape[0]), np.arange(deltas.shape[1])
    ii, jj = np.meshgrid(i, j, indexing='ij')
    index_map = np.stack((ii, jj), axis=-1)

    convergence = deltas + index_map
    convergence = np.pad(convergence, ((1, 1), (1, 1), (0, 0)))
    labels = np.pad(labels, ((1, 1), (1, 1)))

    vmap_count_convergence = vmap(vmap(_count_convergence,
                                       in_axes=(None, None, None, 0)), in_axes=(None, None, 0, None))
    counts = vmap_count_convergence(convergence, labels, i, j)

    return counts


vmap_colocalize_pixels = vmap(colocalize_pixels, in_axes=(0, 0))


def _count_convergence(convergence, labels, i, j):

    convergence = dynamic_slice(convergence, (i, j, 0), (3, 3, 2))
    labels = dynamic_slice(labels, (i, j), (3, 3))
    sources = _search_convergence(convergence, i, j)
    count = np.sum(sources * labels)

    return count


def _search_convergence(convergence, i, j):

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) \
              & (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources
