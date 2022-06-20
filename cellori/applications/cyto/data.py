import glob
import numpy as np
import os

from imageio import imread
from skimage import measure

from cellori.utils.dynamics import masks_to_flows
from cellori.utils.transforms import normalize, RandomAugment


def load_dataset(folder, calculate_gradients=True, use_gpu=True):

    images = []
    masks = []

    image_files = sorted(glob.glob(os.path.join(folder, '*img.png')))
    mask_files = sorted(glob.glob(os.path.join(folder, '*masks.png')))

    for image_file, mask_file in zip(image_files, mask_files):

        image = imread(image_file)[:, :, :2]
        mask = imread(mask_file)

        images.append(image)
        masks.append(mask)

    if calculate_gradients:
        gradients = [np.moveaxis(masks_to_flows(mask, use_gpu=use_gpu), 0, -1) for mask in masks]
    else:
        gradients = None

    ds = {
        'image': images,
        'mask': masks,
        'gradients': gradients
    }

    return ds


def transform_dataset(ds, key, resize_diameter=30, output_shape=(256, 256)):

    # Find median diameters
    diameters = [
        np.median([
            region.equivalent_diameter_area for region in measure.regionprops(mask)
        ]) for mask in ds['mask']
    ]
    base_scales = resize_diameter / np.array(diameters)

    # Create transformer
    transformer = RandomAugment()
    transformer.generate_transforms(ds['image'], key, base_scales, output_shape)

    # Apply transformations
    images = transformer.apply_image_transforms(ds['image'], interpolation='bilinear')
    masks = transformer.apply_image_transforms(ds['mask'], interpolation='nearest')
    gradients = transformer.apply_image_transforms(ds['gradients'], interpolation='bilinear')

    # Normalize images
    images = [normalize(image) for image in images]

    # Transform flow values
    for i, g in enumerate(gradients):

        theta = transformer.theta[i]
        flip0 = transformer.flip0[i]
        flip1 = transformer.flip1[i]

        g = (np.array([[[
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]]]) @ g[:, :, :, None])[:, :, :, 0]

        if flip0:
            g[:, :, 0] *= -1
        if flip1:
            g[:, :, 1] *= -1

        gradients[i] = g

    transformed_ds = {
        'image': np.array(images),
        'mask': np.array(masks)[:, :, :, None],
        'gradients': np.array(gradients),
        'semantic': np.array([mask > 0 for mask in masks])[:, :, :, None]
    }

    return transformed_ds
