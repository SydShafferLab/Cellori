import fastremap
import numpy as np

from scipy import ndimage
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage import transform

from cellori.utils import dynamics


def compute_masks_dynamics(dP, cellprob, p=None, niter=200,
                           cellprob_threshold=0.5, interp=True,
                           min_size=15, resize=None,
                           use_gpu=False, device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary """

    cp_mask = cellprob > cellprob_threshold

    if np.any(cp_mask):  # mask at this point is a cell cluster binary map, not labels
        # follow flows
        if p is None:
            p, inds = dynamics.follow_flows(dP * cp_mask / 5., niter=niter, interp=interp,
                                            use_gpu=use_gpu, device=device)
            if inds is None:
                # dynamics_logger.info('No cell pixels found.')
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p

        # calculate masks
        mask = _compute_masks_dynamics(p, iscell=cp_mask)

        # flow thresholding factored out of get_masks
        # if not do_3D:
        #     # shape0 = p.shape[1:]
        #     if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        #         # make sure labels are unique at output of get_masks
        #         mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device)

        if resize is not None:
            # if verbose:
            #    dynamics_logger.info(f'resizing output with resize = {resize}')
            if mask.max() > 2 ** 16 - 1:
                recast = True
                mask = mask.astype(np.float32)
            else:
                recast = False
                mask = mask.astype(np.uint16)
            if mask.ndim == 2:
                mask = transform.resize(mask, resize)
            elif mask.ndim == 3:
                mask = transform.resize(mask, (mask.shape[0], *resize))
            if recast:
                mask = mask.astype(np.uint32)
            # Ly, Lx = mask.shape
        elif mask.max() < 2 ** 16:
            mask = mask.astype(np.uint16)

    else:  # nothing to compute, just make it compatible
        # dynamics_logger.info('No cell pixels found.')
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p

    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger...
    mask = _clean_mask(mask, min_size=min_size)

    # if mask.dtype == np.uint32:
        # dynamics_logger.warning('more than 65535 masks in image, masks returned as np.uint32')

    return mask, p


def _compute_masks_dynamics(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    inds = None
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing='ij')
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = ndimage.maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    # Nmax = h[seeds]
    # isort = np.argsort(Nmax)[::-1]
    # for s in seeds:
    #     s = s[isort]

    pix = list(np.array(seeds).T)

    # shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    # for e in expand:
    #     e = np.expand_dims(e, 1)

    for iteration in range(5):
        for k in range(len(pix)):
            if iteration == 0:
                pix[k] = list(pix[k])
            newpix = []
            # iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                # iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            # iin = np.all(tuple(iin), axis=0)
            # for p in newpix:
            #     p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iteration == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    # fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0


def _clean_mask(mask, min_size):

    cleaned_mask = np.zeros_like(mask)
    regions = measure.regionprops(mask)

    i = 1
    for region in regions:
        if region.area > min_size:
            filled = ndimage.binary_fill_holes(region.image)
            cleaned_mask[region.slice][filled] = i
            i += 1

    return cleaned_mask


def generate_mask_watershed(distance_transform, class_transform):

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
