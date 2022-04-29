import cv2 as cv
import numpy as np

from functools import partial
from jax import random
from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation


class RandomAffine:

    def __init__(self):

        self.flip0 = []
        self.flip1 = []
        self.scale = []
        self.dxy = []
        self.theta = []
        self.affine_transforms = []

    def generate_transforms(self, images, key, base_scales, output_shape):

        for image, base_scale in zip(images, base_scales):

            # Random flip
            key, *subkeys = random.split(key, 3)
            flip0 = random.uniform(subkeys[0]) > 0.5
            self.flip0.append(flip0)
            flip1 = random.uniform(subkeys[1]) > 0.5
            self.flip1.append(flip1)

            # Random scaling
            key, subkey = random.split(key)
            scale = base_scale * (1 + (random.uniform(subkey) - 0.5) / 2)
            self.scale.append(scale)

            # Random translation
            key, subkey = random.split(key)
            dxy = np.maximum(0, np.array([image.shape[1] * scale - output_shape[1],
                                          image.shape[0] * scale - output_shape[0]]))
            dxy = (random.uniform(subkey, (2,)) - 0.5) * dxy
            self.dxy.append(dxy)

            # Random rotation
            key, subkey = random.split(key)
            theta = random.uniform(subkey) * 2 * np.pi
            self.theta.append(theta)

            # Construct affine transformation
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
            affine = cv.getRotationMatrix2D(image_center, float(theta * 180 / np.pi), float(scale))
            affine[:, 2] += np.array(output_shape) / 2 - np.array(image_center) + dxy
            self.affine_transforms.append(partial(cv.warpAffine, M=affine, dsize=output_shape))

    def apply_transforms(self, images, interpolation='nearest'):

        transformed_images = []

        for image, flip0, flip1, affine_transform in zip(images, self.flip0, self.flip1, self.affine_transforms):

            # Apply affine transformation
            if interpolation == 'nearest':
                transformed_image = affine_transform(image, flags=cv.INTER_NEAREST)
            elif interpolation == 'bilinear':
                transformed_image = affine_transform(image, flags=cv.INTER_LINEAR)

            # Random flip
            if flip0:
                transformed_image = np.flip(transformed_image, axis=0)
            if flip1:
                transformed_image = np.flip(transformed_image, axis=1)

            transformed_images.append(transformed_image)

        return transformed_images


def normalize(image, epsilon=1e-7):

    return (image - np.min(image)) / (np.ptp(image) + epsilon)


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
