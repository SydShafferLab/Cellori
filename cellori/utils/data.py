import cv2 as cv
import glob
import numpy as np
import os

from imageio import imread
from jax import random
from skimage import measure

from cellori.utils import transforms

def load_cellpose_dataset(train, test):

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


def generate_cellpose_dataset(X, y, key, resize_diameter=30, output_shape=(384, 384)):

    dataset = {
        'image': [],
        'distance_transform': [],
        'class_transform': []
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
        theta = random.uniform(subkey) * 2 * np.pi

        # Construct affine transformation
        c0 = np.array(mask.shape) / 2
        cf = np.array(output_shape) / 2 + dxy
        pts1 = np.float32([c0, c0 + np.array([1, 0]), c0 + np.array([0, 1])])
        pts2 = np.float32([cf,
                           cf + scale * np.array([np.cos(theta), np.sin(theta)]),
                           cf + scale * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)])])
        affine = cv.getAffineTransform(pts1, pts2)

        # Apply affine transformation
        image = cv.warpAffine(image, affine, dsize=output_shape, flags=cv.INTER_LINEAR)
        image = (image - np.min(image)) / (np.ptp(image) + 1e-7)
        mask = cv.warpAffine(mask, affine, dsize=output_shape, flags=cv.INTER_NEAREST)
        distance_transform = transforms.distance_transform(mask, alpha='auto')
        class_transform = transforms.class_transform(mask)

        dataset['image'].append(image)
        dataset['distance_transform'].append(distance_transform)
        dataset['class_transform'].append(class_transform)

    dataset['image'] = np.array(dataset['image'])
    dataset['distance_transform'] = np.array(dataset['distance_transform'])[:, :, :, None]
    dataset['class_transform'] = np.array(dataset['class_transform'])

    return dataset
