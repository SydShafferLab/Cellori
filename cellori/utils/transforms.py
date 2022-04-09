import numpy as np

from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.segmentation import find_boundaries


def class_transform(mask, dilation_radius=None, separate_edge_classes=False):

    # Detect the edges and interiors
    edge = find_boundaries(mask, mode='inner').astype('int')
    interior = ((edge == 0) & (mask > 0)).astype(int)

    if separate_edge_classes:

        strel = disk(1)

        # dilate the background masks and subtract from all edges for background-edges
        background = (mask == 0).astype('int')
        dilated_background = binary_dilation(background, strel)

        background_edge = (edge - dilated_background > 0).astype('int')

        # edges that are not background-edges are interior-edges
        interior_edge = (edge - background_edge > 0).astype('int')

        if dilation_radius:

            dil_strel = disk(dilation_radius)

            # Thicken cell edges to be more pronounced
            interior_edge = binary_dilation(interior_edge, footprint=dil_strel)
            background_edge = binary_dilation(background_edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            interior_edge = (interior_edge - interior > 0).astype('int')
            background_edge = (background_edge - interior > 0).astype('int')

        background = (1 - background_edge - interior_edge - interior > 0).astype('int')

        all_stacks = [
            background_edge,
            interior_edge,
            interior,
            background
        ]

    else:

        if dilation_radius:

            dil_strel = disk(dilation_radius)

            # Thicken cell edges to be more pronounced
            edge = binary_dilation(edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            edge = (edge - interior > 0).astype('int')

        background = (1 - edge - interior > 0).astype('int')

        all_stacks = [
            edge,
            interior,
            background
        ]

    return np.stack(all_stacks, axis=-1)


def distance_transform(mask, mode='combined', alpha=0.1, beta=1, bins=None):

    # Check input to alpha
    if isinstance(alpha, str):
        alpha = alpha.lower()
        if alpha != 'auto':
            raise ValueError('alpha must be set to "auto"')

    mask = np.squeeze(mask)
    outer_distance_transform = ndimage.distance_transform_edt(mask)

    distance_transform = np.zeros_like(mask, dtype=float)

    for region in regionprops(mask, outer_distance_transform):

        coords = region.coords
        i, j = coords.T

        distance = None

        if mode == 'outer':

            distance = outer_distance_transform[i, j]

        else:

            center = region.weighted_centroid
            area = region.area

            if mode == 'inner':
                distance = _inner_distance(coords, center, area, alpha, beta)
            elif mode == 'combined':
                distance = outer_distance_transform[i, j] * _inner_distance(coords, center, area, alpha, beta)

        distance_transform[i, j] = distance / distance.max()

    if bins:

        # divide into bins
        min_dist = np.amin(distance_transform.flatten())
        max_dist = np.amax(distance_transform.flatten())
        distance_bins = np.linspace(min_dist - 1e-7,
                                    max_dist + 1e-7,
                                    num=bins + 1)
        distance_transform = np.digitize(distance_transform, distance_bins, right=True)
        distance_transform = distance_transform - 1  # minimum distance should be 0, not 1

    return distance_transform


def _inner_distance(coords, center, area, alpha, beta):

    center_distance = np.sum((coords - center) ** 2, axis=1)

    # Determine alpha to use
    if alpha == 'auto':
        _alpha = 1 / np.sqrt(area)
    else:
        _alpha = float(alpha)

    inner_distance = 1 / (1 + beta * _alpha * center_distance)

    return inner_distance
