import numpy as np

from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation


def class_transform(mask, dilation_radius=None, separate_edge_classes=False):

    # Detect the edges and interiors
    edge = segmentation.find_boundaries(mask, mode='inner').astype('int')
    interior = ((edge == 0) & (mask > 0)).astype(int)

    if separate_edge_classes:

        strel = morphology.disk(1)

        # dilate the background masks and subtract from all edges for background-edges
        background = (mask == 0).astype('int')
        dilated_background = morphology.binary_dilation(background, strel)

        background_edge = (edge - dilated_background > 0).astype('int')

        # edges that are not background-edges are interior-edges
        interior_edge = (edge - background_edge > 0).astype('int')

        if dilation_radius:
            dil_strel = morphology.disk(dilation_radius)

            # Thicken cell edges to be more pronounced
            interior_edge = morphology.binary_dilation(interior_edge, footprint=dil_strel)
            background_edge = morphology.binary_dilation(background_edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            interior_edge = (interior_edge - interior > 0).astype('int')
            background_edge = (background_edge - interior > 0).astype('int')

        background = (1 - background_edge - interior_edge - interior > 0).astype('int')

        all_stacks = [
            background,
            interior,
            interior_edge,
            background_edge
        ]

    else:

        if dilation_radius:
            dil_strel = morphology.disk(dilation_radius)

            # Thicken cell edges to be more pronounced
            edge = morphology.binary_dilation(edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            edge = (edge - interior > 0).astype('int')

        background = (1 - edge - interior > 0).astype('int')

        all_stacks = [
            background,
            interior,
            edge,
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

    transform = np.zeros_like(mask, dtype=float)

    for region in measure.regionprops(mask, outer_distance_transform):

        coords = region.coords
        i, j = coords.T

        distance = None

        if mode == 'outer':

            distance = outer_distance_transform[i, j]

        else:

            if mode == 'inner':
                distance = _inner_distance(region, alpha, beta)
            elif mode == 'combined':
                distance = outer_distance_transform[i, j] * _inner_distance(region, alpha, beta)

        transform[i, j] = distance / distance.max()

    transform = filters.gaussian(transform)
    transform = filters.unsharp_mask(transform)
    transform = transform / (transform.max() + 1e-7)

    if bins:
        # divide into bins
        min_dist = np.amin(transform.flatten())
        max_dist = np.amax(transform.flatten())
        distance_bins = np.linspace(min_dist - 1e-7,
                                    max_dist + 1e-7,
                                    num=bins + 1)
        transform = np.digitize(transform, distance_bins, right=True)
        transform = transform - 1  # minimum distance should be 0, not 1

    return transform


def _inner_distance(region, alpha, beta):

    center_distance = np.sum((region.coords - region.weighted_centroid) ** 2, axis=1)

    # Determine alpha to use
    if alpha == 'auto':
        _alpha = 2 / region.equivalent_diameter_area
    else:
        _alpha = float(alpha)

    inner_distance = 1 / (1 + beta * _alpha * center_distance)

    return inner_distance
