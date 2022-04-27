import cv2 as cv
import glob
import numpy as np
import os

from imageio import imread
from jax import random
from skimage import measure

from cellori.utils import dynamics

def load_dataset(train, test):

    def load_folder(folder):

        X = []
        y = []

        image_files = sorted(glob.glob(os.path.join(folder, '*img.png')))
        mask_files = sorted(glob.glob(os.path.join(folder, '*masks.png')))

        for image_file, mask_file in zip(image_files, mask_files):

            image = imread(image_file)[:, :, :2]
            mask = imread(mask_file)

            X.append(image)
            y.append(mask)

        return X, y

    X_train, y_train = load_folder(train)
    X_test, y_test = load_folder(test)

    return X_train, y_train, X_test, y_test


def generate_dataset(X, y, key, resize_diameter=30, output_shape=(256, 256)):

    dataset = {
        'image': [],
        'gradients': [],
        'semantic': []
    }

    for image, mask in zip(X, y):

        # Find median diameter
        diameter = np.median([region.equivalent_diameter_area for region in measure.regionprops(mask)])

        # Random flip
        key, subkey = random.split(key)
        if random.uniform(subkey) > 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        key, subkey = random.split(key)
        if random.uniform(subkey) > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        # Random scaling
        key, subkey = random.split(key)
        scale = 1 + (random.uniform(subkey) - 0.5) / 2
        scale = (resize_diameter / diameter) * scale

        # Random translation
        key, subkey = random.split(key)
        dxy = np.maximum(0, np.array([mask.shape[1] * scale - output_shape[1],
                                      mask.shape[0] * scale - output_shape[0]]))
        dxy = (random.uniform(subkey, (2, )) - 0.5) * dxy

        # Random rotation
        key, subkey = random.split(key)
        theta = random.uniform(subkey) * 360

        # Construct affine transformation
        image_center = (mask.shape[1] / 2, mask.shape[0] / 2)
        affine = cv.getRotationMatrix2D(image_center, float(theta), float(scale))
        affine[:, 2] += np.array(output_shape) / 2 - np.array(image_center) + dxy

        # Apply affine transformation
        image = cv.warpAffine(image, affine, dsize=output_shape, flags=cv.INTER_LINEAR)
        image = (image - np.min(image)) / (np.ptp(image) + 1e-7)
        mask = cv.warpAffine(mask, affine, dsize=output_shape, flags=cv.INTER_NEAREST)
        gradients = np.moveaxis(dynamics.masks_to_flows(mask), 0, 2)

        dataset['image'].append(image)
        dataset['gradients'].append(gradients)
        dataset['semantic'].append(mask > 0)

    dataset['image'] = np.array(dataset['image'])
    dataset['gradients'] = np.array(dataset['gradients'])
    dataset['semantic'] = np.array(dataset['semantic'])[:, :, :, None]

    return dataset
