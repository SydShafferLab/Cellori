import fastremap
import numpy as np
import os
import torch
import tifffile

from numba import njit
from skimage import measure
from tqdm import trange


@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                                  T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                                  T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                                  T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])
    return T


def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control

    neighbors is 9 x pixels in masks,
    centers are mask centers,
    isneighbor is valid neighbor boolean 9 x pixels

    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)

    T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)

    T = torch.log(1. + T)
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]

    mu_torch = np.stack((dy[0].cpu(), dx[0].cpu()), axis=-2)
    return mu_torch


def masks_to_flows_gpu(masks):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides
    """

    masks = measure.label(masks)

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1,
                           y, y, y - 1,
                           y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x,
                           x - 1, x + 1, x - 1,
                           x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    regions = measure.regionprops(masks)
    slices = [region.slice for region in regions]
    centers = np.zeros((masks.max(), 2), 'int')

    for i, si in enumerate(slices):

        sr, sc = si
        yi, xi = np.where(masks[sr, sc] == (i + 1))
        ymed = np.median(yi)
        xmed = np.median(xi)
        imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
        xmed = xi[imin]
        ymed = yi[imin]
        centers[i, 0] = ymed + sr.start
        centers[i, 1] = xmed + sc.start

    centers += 1  # add padding

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx,
                             n_iter=n_iter)

    # normalize
    mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y - 1, x - 1] = mu
    return mu0


def masks_to_flows_cpu(masks):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.
    Parameters
    -------------
    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask
        in which it resides
    """

    masks = measure.label(masks)

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)

    # get mask centers
    regions = measure.regionprops(masks)
    slices = [region.slice for region in regions]

    for i, si in enumerate(slices):

        sr, sc = si
        ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
        y, x = np.nonzero(masks[sr, sc] == (i + 1))
        y = y.astype(np.int32) + 1
        x = x.astype(np.int32) + 1

        ymed = np.median(y)
        xmed = np.median(x)
        imin = np.argmin((x - xmed) ** 2 + (y - ymed) ** 2)
        xmed = x[imin]
        ymed = y[imin]

        niter = 2 * np.int32(np.ptp(x) + np.ptp(y))
        T = np.zeros((ly + 2) * (lx + 2), np.float64)
        T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(niter))
        T[(y + 1) * lx + x + 1] = np.log(1. + T[(y + 1) * lx + x + 1])

        dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
        dx = T[y * lx + x + 1] - T[y * lx + x - 1]
        mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))

    mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)

    return mu


def masks_to_flows(masks, use_gpu=False):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """
    if masks.max() == 0:
        # dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), 'float32')

    if use_gpu:
        masks_to_flows_device = masks_to_flows_gpu
    else:
        masks_to_flows_device = masks_to_flows_cpu

    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            masksz = masks[z]
            if masksz.max() > 0:
                mu0 = masks_to_flows_device(masksz)[0]
                mu[[1, 2], z] += mu0
        for y in range(Ly):
            masksy = masks[:, y]
            if masksy.max() > 0:
                mu0 = masks_to_flows_device(masksy)[0]
                mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            masksx = masks[:, :, x]
            if masksx.max() > 0:
                mu0 = masks_to_flows_device(masksx)[0]
                mu[[0, 1], :, :, x] += mu0
        return mu
    elif masks.ndim == 2:
        mu = masks_to_flows_device(masks)
        return mu

    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')


def labels_to_flows(labels, files=None, use_gpu=False, redo_flows=False):
    """ convert labels (list of masks or flows) to flows for training model

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2] is Y flow,
        flows[k][3] is X flow, and flows[k][4] is heat distribution

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows:  # flows need to be recomputed

        # dynamics_logger.info('computing flows for labels')

        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [masks_to_flows(labels[n][0], use_gpu=use_gpu) for n in trange(nimg)]

        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(np.float32)
                 for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imwrite(file_name + '_flows.tif', flow)
    else:
        # dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])',
       '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y

    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C, Ly, Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly - 1, max(0, yc_floor[i]))
        xf = min(Lx - 1, max(0, xc_floor[i]))
        yf1 = min(Ly - 1, yf + 1)
        xf1 = min(Lx - 1, xf + 1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c, i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                       np.float32(I[c, yf, xf1]) * (1 - y) * x +
                       np.float32(I[c, yf1, xf]) * y * (1 - x) +
                       np.float32(I[c, yf1, xf1]) * y * x)


def steps2D_interp(p, dP, niter, use_gpu=False, device=torch.device('cuda')):
    shape = dP.shape[1:]
    if use_gpu:
        shape = np.array(shape)[[1, 0]].astype('float') - 1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1, 0]].T).float().to(device).unsqueeze(0).unsqueeze(
            0)  # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1, 0]]).float().to(device).unsqueeze(
            0)  # covert flow numpy array to tensor on GPU, add dimension
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2):
            im[:, k, :, :] *= 2. / shape[k]
            pt[:, :, :, k] /= shape[k]

        # normalize to between -1 and 1
        pt = pt * 2 - 1

        # here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)

            for k in range(2):  # clamp the final pixel locations
                pt[:, :, :, k] = torch.clamp(pt[:, :, :, k] + dPt[:, k, :, :], -1., 1.)

        # undo the normalization from before, reverse order of operations
        pt = (pt + 1) * 0.5
        for k in range(2):
            pt[:, :, :, k] *= shape[k]

        p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T
        return p

    else:
        dPt = np.zeros(p.shape, np.float32)

        for t in range(niter):
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k] - 1, np.maximum(0, p[k] + dPt[k]))
        return p


@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D

    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 4D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        # pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j, 0]
            y = inds[j, 1]
            x = inds[j, 2]
            p0, p1, p2 = int(p[0, z, y, x]), int(p[1, z, y, x]), int(p[2, z, y, x])
            p[0, z, y, x] = min(shape[0] - 1, max(0, p[0, z, y, x] + dP[0, p0, p1, p2]))
            p[1, z, y, x] = min(shape[1] - 1, max(0, p[1, z, y, x] + dP[1, p0, p1, p2]))
            p[2, z, y, x] = min(shape[2] - 1, max(0, p[2, z, y, x] + dP[2, p0, p1, p2]))
    return p


@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D

    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]
            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))
    return p


def follow_flows(dP, niter=200, interp=True, use_gpu=True, device=None):
    """ define pixels and run dynamics to recover masks in 2D

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D)
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    if len(shape) > 2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                        np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)

        inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

        if inds.ndim < 2 or inds.shape[0] < 5:
            # dynamics_logger.warning('WARNING: no mask pixels found')
            return p, None

        if not interp:
            p = steps2D(p, dP.astype(np.float32), inds, niter)

        else:
            p_interp = steps2D_interp(p[:, inds[:, 0], inds[:, 1]], dP, niter, use_gpu=use_gpu, device=device)
            p[:, inds[:, 0], inds[:, 1]] = p_interp
    return p, inds
