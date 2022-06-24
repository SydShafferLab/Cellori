import cv2 as cv
import jax.numpy as np
import numpy as onp

from jax import jit, random, vmap
from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation


class RandomAugment:

    def __init__(self):

        self.output_shape = None
        self.flip0 = []
        self.flip1 = []
        self.scale = []
        self.dxy = []
        self.theta = []
        self.affines = []

    def generate_transforms(self, images, key, base_scales, output_shape):

        # Reset
        self.output_shape = output_shape
        self.flip0 = []
        self.flip1 = []
        self.scale = []
        self.dxy = []
        self.theta = []
        self.affines = []

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
            dxy = onp.maximum(0, onp.array([image.shape[1] * scale - output_shape[1],
                                            image.shape[0] * scale - output_shape[0]]))
            dxy = (random.uniform(subkey, (2,)) - 0.5) * dxy
            self.dxy.append(dxy)

            # Random rotation
            key, subkey = random.split(key)
            theta = random.uniform(subkey) * 2 * onp.pi
            self.theta.append(theta)

            # Construct affine transformation
            image_center = (image.shape[1] / 2, image.shape[0] / 2)
            affine = cv.getRotationMatrix2D(image_center, float(theta * 180 / onp.pi), float(scale))
            affine[:, 2] += onp.array(output_shape) / 2 - onp.array(image_center) + dxy
            self.affines.append(affine)

    def apply_coord_transforms(self, coords, filter_coords=True):

        transformed_coords = []

        for coord, flip0, flip1, affine in zip(coords, self.flip0, self.flip1, self.affines):

            # Apply affine transformation
            coord = onp.concatenate((onp.flip(coord, axis=1), onp.ones((len(coord), 1))), axis=1)
            transformed_coord = onp.flip((coord @ affine.T), axis=1)

            # Random flip
            if flip0:
                transformed_coord[:, 0] = self.output_shape[0] - 0.5 - transformed_coord[:, 0]
            if flip1:
                transformed_coord[:, 1] = self.output_shape[1] - 0.5 - transformed_coord[:, 1]

            # Filter coordinates outside of transformed image
            if filter_coords:
                transformed_coord = transformed_coord[onp.all((transformed_coord > -0.5) &
                                                              (transformed_coord < onp.array(self.output_shape) - 0.5),
                                                              axis=1)]

            transformed_coords.append(transformed_coord)

        return transformed_coords

    def apply_image_transforms(self, images, interpolation='nearest'):

        transformed_images = []

        for image, flip0, flip1, affine in zip(images, self.flip0, self.flip1, self.affines):

            # Apply affine transformation
            if interpolation == 'nearest':
                transformed_image = cv.warpAffine(image, M=affine, dsize=self.output_shape, flags=cv.INTER_NEAREST)
            elif interpolation == 'bilinear':
                transformed_image = cv.warpAffine(image, M=affine, dsize=self.output_shape, flags=cv.INTER_LINEAR)

            # Random flip
            if flip0:
                transformed_image = onp.flip(transformed_image, axis=0)
            if flip1:
                transformed_image = onp.flip(transformed_image, axis=1)

            transformed_images.append(transformed_image)

        return transformed_images


def normalize(image, lower=0, upper=100, epsilon=1e-7):

    image_lower = onp.percentile(image, lower)
    image_upper = onp.percentile(image, upper)

    return (image - image_lower) / onp.maximum(image_upper - image_lower, epsilon)


def batch_normalize(images, lower=0, upper=100, epsilon=1e-7):

    return vmap(_normalize, in_axes=(0, None, None, None))(images, lower, upper, epsilon)


@jit
def _normalize(image, lower, upper, epsilon):

    image_lower = np.percentile(image, lower)
    image_upper = np.percentile(image, upper)

    return (image - image_lower) / np.maximum(image_upper - image_lower, epsilon)


def batch_standardize(images, epsilon=1e-7):

    return vmap(_standardize, in_axes=(0, None))(images, epsilon)


@jit
def _standardize(image, epsilon):

    return (image - np.mean(image)) / np.maximum(np.std(image), epsilon)


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

    return onp.stack(all_stacks, axis=-1)


def distance_transform(mask, mode='combined', alpha=0.1, beta=1, bins=None):

    # Check input to alpha
    if isinstance(alpha, str):
        alpha = alpha.lower()
        if alpha != 'auto':
            raise ValueError('alpha must be set to "auto"')

    mask = onp.squeeze(mask)
    outer_distance_transform = ndimage.distance_transform_edt(mask)

    transform = onp.zeros_like(mask, dtype=float)

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
        min_dist = onp.amin(transform.flatten())
        max_dist = onp.amax(transform.flatten())
        distance_bins = onp.linspace(min_dist - 1e-7,
                                     max_dist + 1e-7,
                                     num=bins + 1)
        transform = onp.digitize(transform, distance_bins, right=True)
        transform = transform - 1  # minimum distance should be 0, not 1

    return transform


def _inner_distance(region, alpha, beta):

    center_distance = onp.sum((region.coords - region.weighted_centroid) ** 2, axis=1)

    # Determine alpha to use
    if alpha == 'auto':
        _alpha = 2 / region.equivalent_diameter_area
    else:
        _alpha = float(alpha)

    inner_distance = 1 / (1 + beta * _alpha * center_distance)

    return inner_distance


def subpixel_distance_transform(coords_list, coords_pad_length=None, shape=(128, 128), dy=1, dx=1):

    """For each pixel in an image, return the vertical and horizontal distances to a point in
    ``coords`` that is in the pixel nearest to it.
    """

    batch_size = len(coords_list)
    labels = onp.zeros((batch_size, ) + shape, dtype=bool)

    max_num_coords = onp.max([len(coords) for coords in coords_list])
    if (coords_pad_length is None) or (coords_pad_length < max_num_coords):
        coords_pad_length = max_num_coords

    subpixel_coords = onp.zeros((batch_size, coords_pad_length, 2))
    rounded_coords = onp.zeros((batch_size, coords_pad_length, 2), dtype=int)
    edt_indices = onp.zeros((batch_size, 2) + shape, dtype=int)

    for i, coords in enumerate(coords_list):

        padding = ((0, coords_pad_length - len(coords)), (0, 0))
        subpixel_coords[i] = onp.pad(coords, padding, constant_values=-1)
        unpadded_rounded_coords = onp.rint(coords).astype(int)
        labels[i][unpadded_rounded_coords[:, 0], unpadded_rounded_coords[:, 1]] = True
        rounded_coords[i] = onp.pad(unpadded_rounded_coords, padding, constant_values=-1)
        edt_indices[i] = ndimage.distance_transform_edt(~labels[i], return_distances=False, return_indices=True,
                                                        sampling=(dy, dx))

    labels = np.expand_dims(labels, -1)
    subpixel_coords = np.array(subpixel_coords)
    rounded_coords = np.array(rounded_coords)
    edt_indices = np.array(edt_indices)

    inputs = [subpixel_coords, rounded_coords, edt_indices, dy, dx]
    vmap_vmap_sdt = vmap(_vmap_sdt, in_axes=([0, 0, 0, None, None], None, None))
    deltas, nearest = vmap_vmap_sdt(inputs, np.arange(shape[0]), np.arange(shape[1]))

    return deltas, labels, nearest


def _vmap_sdt(inputs, i, j):

    return vmap(vmap(_sdt, in_axes=(None, None, 0)), in_axes=(None, 0, None))(inputs, i, j)


@jit
def _sdt(inputs, i, j):

    subpixel_coords, rounded_coords, edt_indices, dy, dx = inputs
    nearest = np.where(np.all(rounded_coords == edt_indices[:, i, j], axis=1)[:, None], subpixel_coords, 0)
    nearest = np.sum(nearest, axis=0)
    delta_y = dy * (nearest[0] - i)
    delta_x = dx * (nearest[1] - j)
    delta = np.stack((delta_y, delta_x), axis=-1)

    return delta, nearest
