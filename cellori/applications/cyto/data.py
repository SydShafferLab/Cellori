import glob
import numpy as np
import os

from imageio import imread
from skimage import measure

from cellori.utils.dynamics import masks_to_flows
from cellori.utils.transforms import normalize, RandomAffine

def load_dataset(train, test, use_gpu=True):

    def load_folder(folder):

        images = []
        masks = []

        image_files = sorted(glob.glob(os.path.join(folder, '*img.png')))
        mask_files = sorted(glob.glob(os.path.join(folder, '*masks.png')))

        for image_file, mask_file in zip(image_files, mask_files):

            image = imread(image_file)[:, :, :2]
            mask = imread(mask_file)

            images.append(image)
            masks.append(mask)

        return images, masks

    images_train, masks_train = load_folder(train)
    images_test, masks_test = load_folder(test)

    gradients_train = [np.moveaxis(masks_to_flows(mask, use_gpu=use_gpu), 0, -1) for mask in masks_train]
    gradients_test = [np.moveaxis(masks_to_flows(mask, use_gpu=use_gpu), 0, -1) for mask in masks_test]

    train_ds = {
        'image': images_train,
        'mask': masks_train,
        'gradients': gradients_train
    }

    test_ds = {
        'image': images_test,
        'mask': masks_test,
        'gradients': gradients_test
    }

    return train_ds, test_ds


def transform_dataset(ds, key, resize_diameter=30, output_shape=(256, 256)):

    # Find median diameters
    diameters = [
        np.median([
            region.equivalent_diameter_area for region in measure.regionprops(mask)
        ]) for mask in ds['mask']
    ]
    base_scales = resize_diameter / np.array(diameters)

    # Create transformer
    transformer = RandomAffine()
    transformer.generate_transforms(ds['image'], key, base_scales, output_shape)

    # Apply transformations
    images = transformer.apply_transforms(ds['image'], interpolation='bilinear')
    masks = transformer.apply_transforms(ds['mask'], interpolation='nearest')
    gradients = transformer.apply_transforms(ds['gradients'], interpolation='bilinear')

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
        'gradients': np.array(gradients),
        'semantic': np.array([mask > 0 for mask in masks])[:, :, :, None]
    }

    return transformed_ds
