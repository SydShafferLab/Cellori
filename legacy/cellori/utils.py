import cv2 as cv
import numpy as np

from cellori.netmap import get_touch_map, ncolor_label, render_net
from skimage import exposure, measure


def overlay_segmentation(image, masks, overlay_mode='both', mask_alpha=0.5, mask_ncolors=5, sinebow_theta=1,
                         outline_color=None, contrast_z=5, gamma=1):
    """Overlay segmentation results on the original image.

    Parameters
    ----------
        image : numpy.ndarray
            Array of the image segmented by the Cellori algorithm.
        masks : numpy.ndarray
            Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ...,
            N.
        overlay_mode : {'both', 'masks', 'outlines'}, optional, default 'both'
            * ‘both’: Overlay both masks and outlines.
            * ‘masks’: Overlay masks only.
            * ‘outlines’: Overlay outlines only.
        mask_alpha : float, optional, default 0.5
            Alpha value of the mask overlay. The value must be between 0 (transparent) and 1 (opaque).
        mask_ncolors : int, optional, default 5
            Number of colors used for coloring the mask overlay.
        sinebow_theta : float, optional, default 1
            Phase factor in radians for color generation using sinebow.
        outline_color : tuple, optional, default None
            Color of outlines in RGB, formatted as (r,g,b).
        contrast_z : float, optional, default 5
            Number of standard deviations from the mean used for intensity rescaling.
        gamma : float, optional, default 1
            Gamma value used for brightness adjustment.
    
    Returns
    -------
        image : numpy.ndarray
            Array of the same size as the original image with overlayed segmentation results.
    """

    if overlay_mode not in ['both', 'masks', 'outlines']:
        raise ValueError("Invalid overlay mode.")

    if gamma != 1:
        image = image ** gamma
    image = exposure.rescale_intensity(image, (0, np.mean(image) + contrast_z * np.std(image)), (0, 1))
    image = np.repeat(image, 3).reshape(*image.shape, 3)

    if overlay_mode != 'outlines':
        idx = get_touch_map(masks)
        colors = render_net(idx, 4, 10)
        color_label = ncolor_label(masks, colors)
        masks_rgb = _relabel(masks, color_label)
        masks_rgb = _masks_to_rgb(masks_rgb, mask_ncolors, sinebow_theta)
        masks_nonzero = masks > 0
        image[masks_nonzero] = image[masks_nonzero] * (1 - mask_alpha) + masks_rgb[masks_nonzero] * mask_alpha

    image = (image * 255).astype(np.uint8)

    if overlay_mode != 'masks':

        if outline_color is None:
            outline_color = (255, 255, 255) if overlay_mode == 'both' else (0, 0, 255)

        image[_masks_to_outlines(masks)] = outline_color

    return image


def _masks_to_rgb(masks, ncolors, theta):
    """(For internal use) Convert masks to RGB image.

    Parameters
    ----------
        masks : numpy.ndarray
            Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ...,
            N.
    
    Returns
    -------
        masks : numpy.ndarray
            Relabeled mask array with consecutive integer assignments for clustered regions.
    """

    color_map = _sinebow(ncolors, theta)
    colors = (masks % ncolors + 1) * (masks > 0)
    regions = 3 * colors[colors >= 1]
    regions = np.repeat(regions, 3).reshape(-1, 3) - np.array([2, 1, 0])
    masks = np.expand_dims(masks, -1)
    masks = np.repeat(masks, 3, -1).astype(float)
    masks[masks > 0] = _vector_map(regions, color_map).flatten()

    return masks


def _masks_to_outlines(masks):
    """(For internal use) Convert masks to outlines for displaying GUI segmentation results.

    Parameters
    ----------
        masks : numpy.ndarray
            Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ...,
            N.
    
    Returns
    -------
        outlines : numpy.ndarray
            Array of the same size as the original image with outlines around each cell.
    """

    outlines = np.zeros(masks.shape, dtype=bool)

    regions = measure.regionprops(masks, cache=False)
    for region in regions:
        mask = region.image.astype(np.uint8)
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours[0], axis=0).squeeze().T
        outlines[contours[1] + region.slice[0].start, contours[0] + region.slice[1].start] = 1

    return outlines


def _relabel(masks, color_label):
    """(For internal use) Relabel mask array by assigning consecutive integers to clustered regions.

    Parameters
    ----------
        masks : numpy.ndarray
            Labeled array of the same size as the original image with background pixels as 0 and cells as 1, 2, 3, ...,
            N.
    
    Returns
    -------
        masks : numpy.ndarray
            Relabeled mask array with consecutive integer assignments for clustered regions.
    """

    masks = measure.label(masks > 0)
    masks = masks + color_label / (np.max(color_label) + 1)
    regions = masks[masks > 0]
    float_ids = np.unique(regions)
    int_ids = np.argsort(float_ids) + 1
    id_map = dict(zip(float_ids, int_ids))
    masks[masks > 0] = _vector_map(regions, id_map)
    masks = masks.astype(int)

    return masks


def _sinebow(ncolors, theta):
    """(For internal use) Generate n-color dictionary using the sinebow algorithm.

    Parameters
    ----------
        ncolors : int
            Number of colors to generate.
        theta : float
            Phase factor in radians for color generation.
    
    Returns
    -------
        color_dict : dict
            Color dictionary.
    """

    color_dict = dict()

    for n in range(ncolors):
        angle = 2 * np.pi * n / ncolors + theta
        r = ((np.sin(angle)) / 2 + 0.5)
        g = ((np.sin(angle + 2 * np.pi / 3)) / 2 + 0.5)
        b = ((np.sin(angle + 4 * np.pi / 3)) / 2 + 0.5)
        color_dict.update({3 * n + 1: r})
        color_dict.update({3 * n + 2: g})
        color_dict.update({3 * n + 3: b})

    return color_dict


def _vector_map(array, array_map):
    """(For internal use) Vectorized implementation for applying a map to an array.

    Parameters
    ----------
        array : numpy.ndarray
            Array input.
        array_map : dict
            Map to be applied on the array.
    
    Returns
    -------
        array : numpy.ndarray
            Array output.
    """

    array = np.vectorize(array_map.__getitem__)(array)

    return array
