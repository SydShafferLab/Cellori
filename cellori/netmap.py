import numpy as np

from numba import njit
from scipy.ndimage import generate_binary_structure


def ncolor_label(masks, net, clusters_only=True):
    lut = np.ones(masks.max() + 1, dtype=np.uint8)
    for i in net:
        lut[i] = net[i] + clusters_only
    lut[0] = 0
    color_label = lut[masks]

    if clusters_only:
        color_label = np.where(color_label > 1, color_label - 1, 0)

    return color_label


def get_touch_map(masks):
    idx = _connect(masks, 4)
    idx = _mapidx(idx)

    return idx


def _neighbors(shape, conn=1):
    dim = len(shape)
    block = generate_binary_structure(dim, conn)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])

    return np.dot(idx, acc[::-1])


@njit
def _search(img, nbs):
    s, line = 0, img.ravel()
    rst = np.zeros((len(line), 2), img.dtype)
    for i in range(len(line)):
        if line[i] == 0:
            continue
        for d in nbs:
            if line[i + d] == 0:
                continue
            if line[i] == line[i + d]:
                continue
            rst[s, 0] = line[i]
            rst[s, 1] = line[i + d]
            s += 1

    return rst[:s]


def _connect(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = _neighbors(buf.shape, conn)
    rst = _search(buf, nbs)
    if len(rst) < 2:
        return rst
    rst.sort(axis=1)
    key = (rst[:, 0] << 16)
    key += rst[:, 1]
    order = np.argsort(key)
    key[:] = key[order]
    diff = key[:-1] != key[1:]
    idx = np.where(diff)[0] + 1
    idx = np.hstack(([0], idx))

    return rst[order][idx]


def _mapidx(idx):
    dic = {}
    for i in np.unique(idx):
        dic[i] = []
    for i, j in idx:
        dic[i].append(j)
        dic[j].append(i)

    return dic


def render_net(con_map, n=4, rand=12, shuffle=False):
    nodes = list(con_map.keys())
    colors = dict(zip(nodes, [0] * len(nodes)))
    counter = dict(zip(nodes, [0] * len(nodes)))
    if shuffle:
        np.random.shuffle(nodes)
    while len(nodes) > 0:
        k = nodes.pop(0)
        counter[k] += 1
        hist = [1e4] + [0] * n
        for p in con_map[k]:
            hist[colors[p]] += 1
        if min(hist) == 0:
            colors[k] = hist.index(min(hist))
            counter[k] = 0
            continue
        hist[colors[k]] = 1e4
        minc = hist.index(min(hist))
        if counter[k] == rand:
            counter[k] = 0
            minc = 5
        colors[k] = minc
        for p in con_map[k]:
            if colors[p] == minc:
                nodes.append(p)

    return colors
