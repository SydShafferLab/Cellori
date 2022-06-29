from skimage import feature, filters


def log_filter(image, sigma):

    image = filters.gaussian(image, sigma)
    image = filters.laplace(image)

    return image


def threshold_local_max(image, min_distance, threshold):

    coords = feature.peak_local_max(image, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)

    return coords
