import numpy as np

from skimage import feature
from skimage import measure
from skimage import segmentation


def generate_mask(distance_transform, class_transform):

    distance_transform = np.squeeze(distance_transform)
    class_transform = np.array(class_transform)

    # Find local maxima
    coords = feature.peak_local_max(distance_transform, min_distance=3, threshold_abs=0.25, exclude_border=False)

    # Reverse class transform one-hot encoding
    class_transform = np.argmax(class_transform, axis=-1)

    # Generate markers
    markers = np.zeros(distance_transform.shape, dtype=bool)
    markers[tuple(np.rint(coords).astype(int).T)] = True
    markers = measure.label(markers)

    # Run watershed algorithm
    mask = segmentation.watershed(class_transform, markers, mask=class_transform > 0, watershed_line=True)
    mask = measure.label(mask)

    return mask
