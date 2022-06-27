import jax.numpy as np
import numpy as onp

from jax import jit, vmap
from jax.lax import scan
from skimage import feature


def compute_spot_coordinates(deltas, labels, min_distance=1, threshold=0.5):

    coords = []

    for i in range(len(deltas)):

        counts, convergence = jit(colocalize_pixels)(deltas[i], labels[i])
        peaks = feature.peak_local_max(onp.asarray(counts + labels[i, :, :, 0] - 1),
                                       min_distance=min_distance, threshold_abs=threshold)
        num_peaks = len(peaks)
        if num_peaks > 0:
            _, frame_coords = scan(jit(compute_subpixel_coords), (counts, convergence, peaks), np.arange(num_peaks))
            coords.append(onp.array(frame_coords))
        else:
            coords.append(onp.empty((0, 2)))

    return coords


def compute_subpixel_coords(carry, i):

    counts, convergence, peaks = carry
    peak = peaks[i]
    sources = _search_convergence(convergence, peak[0], peak[1])
    coords = np.sum(convergence * sources[:, :, None], axis=(0, 1)) / counts[peak[0], peak[1]]

    return carry, coords


def colocalize_pixels(deltas, labels):

    deltas = labels * deltas
    counts, convergence = follow_deltas(deltas)

    return counts, convergence


def follow_deltas(deltas):

    i, j = np.arange(deltas.shape[0]), np.arange(deltas.shape[1])
    ii, jj = np.meshgrid(i, j, indexing='ij')
    index_map = np.stack((ii, jj), axis=-1)

    convergence = deltas + index_map

    vmap_count_convergence = vmap(vmap(_count_convergence, in_axes=(None, None, 0)), in_axes=(None, 0, None))
    counts = vmap_count_convergence(convergence, i, j)

    return counts, convergence


def _count_convergence(convergence, i, j):

    sources = _search_convergence(convergence, i, j)
    count = np.sum(sources)

    return count


def _search_convergence(convergence, i, j):

    sources = (i - 0.5 < convergence[:, :, 0]) & (convergence[:, :, 0] < i + 0.5) \
              & (j - 0.5 < convergence[:, :, 1]) & (convergence[:, :, 1] < j + 0.5)

    return sources
