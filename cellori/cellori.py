import deeptile
import jax.numpy as np
import numpy as onp

from deeptile import partial, transform
from deeptile.extensions import stitch
from flax.training import checkpoints
from jax import jit
from pathlib import Path
from skimage.transform import resize

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class CelloriSegmentation:

    def __new__(cls, model='cyto', batch_size=None):

        if model == 'cyto':
            model = super(CelloriSegmentation, _CelloriCyto).__new__(_CelloriCyto)

        return model

    @staticmethod
    def preprocess(x, diameter):

        shape = x.shape
        ndim = x.ndim

        if diameter != 30:
            x = resize(x, (*shape[:-2], round(shape[-2] * 30 / diameter), round(shape[-1] * 30 / diameter)))

        if ndim == 3:
            batch_axis = False
            x = (x - onp.min(x)) / (onp.ptp(x) + 1e-7)
            x = onp.pad(x, ((0, 0), (2, 2), (2, 2)))
        elif ndim == 4:
            batch_axis = True
            x = (x - onp.min(x, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))) / \
                (onp.ptp(x, axis=(1, 2, 3)).reshape((-1, 1, 1, 1)) + 1e-7)
            x = onp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))
        else:
            raise ValueError("Input does not have the correct dimensions.")

        return x, shape, batch_axis


class _CelloriCyto(CelloriSegmentation):

    def __init__(self, model='cyto', batch_size=8):

        from cellori.applications.cyto.model import CelloriCytoModel
        from cellori.utils import masks

        self.model_name = model
        self.model = CelloriCytoModel()
        self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath('cyto'), None)
        self.batch_size = batch_size

        @jit
        def jitted(x):

            x = np.moveaxis(x, 1, -1)
            flows, semantic = self.model.apply(self.variables, x, False)
            y = np.concatenate((flows, semantic), axis=-1)
            y = np.moveaxis(y, -1, 1)

            return y

        def process(tiles):

            y = jitted(tiles)
            y = onp.asarray(y)

            return y

        process(np.zeros((self.batch_size, 2, 256, 256)))
        self.process = transform(process)

        def postprocess(tile, cellprob_threshold, flow_threshold):

            flows = onp.array(tile[:2])
            semantic = onp.array(tile[2])
            mask, _ = masks.compute_masks_dynamics(flows, semantic, cellprob_threshold=cellprob_threshold,
                                                   flow_threshold=flow_threshold)

            return mask

        self.postprocess = transform(postprocess)

    def predict(self, x, diameter=30, cellprob_threshold=0.5, flow_threshold=0.5):

        x, shape, batch_axis = self.preprocess(x, diameter)

        dt = deeptile.load(x)
        dt.link_data = False
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).compute()
        tiles = dt.process(tiles, self.process, batch_size=self.batch_size, batch_axis=batch_axis, pad_final_batch=True)
        y = dt.stitch(tiles, stitch.stitch_tiles(blend=True, sigma=5))[..., 2:-2, 2:-2]
        y = resize(y, output_shape=(*y.shape[:-2], shape[-2], shape[-1]), order=1, preserve_range=True)

        dt2 = deeptile.load(y)
        dt.link_data = False
        tile_size2 = (min(1024, y.shape[-2]), min(1024, y.shape[-1]))
        tiles2 = dt2.get_tiles(tile_size=tile_size2, overlap=(0.2, 0.2)).compute()
        tiles2 = dt2.process(tiles2, partial(self.postprocess, cellprob_threshold=cellprob_threshold,
                                             flow_threshold=flow_threshold), batch_axis=batch_axis)
        mask = dt2.stitch(tiles2, stitch.stitch_masks())

        return mask, y


class CelloriSpots:

    def __new__(cls, model='spots', batch_size=None):

        if model == 'spots':
            model = super(CelloriSpots, _CelloriSpots).__new__(_CelloriSpots)
        elif model == 'LoG':
            model = super(CelloriSpots, _CelloriLoG).__new__(_CelloriLoG)

        return model

    @staticmethod
    def preprocess(x, scale):

        shape = x.shape
        ndim = x.ndim

        if scale != 1:
            x = resize(x, (*shape[:-2], round(shape[-2] * scale), round(shape[-1] * scale)))

        if ndim == 2:
            batch_axis = False
            x = (x - onp.min(x)) / (onp.ptp(x) + 1e-7)
        elif ndim == 3:
            batch_axis = True
            x = (x - onp.min(x, axis=(1, 2)).reshape((-1, 1, 1))) / \
                (onp.ptp(x, axis=(1, 2)).reshape((-1, 1, 1)) + 1e-7)
        else:
            raise ValueError("Input does not have the correct dimensions.")

        return x, shape, batch_axis


class _CelloriSpots(CelloriSpots):

    def __init__(self, model, batch_size=8):

        from cellori.applications.spots.model import CelloriSpotsModel
        from cellori.utils import spots

        self.model_name = model
        self.model = CelloriSpotsModel()
        self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath('spots'), None)
        self.batch_size = batch_size

        @jit
        def jitted(x):

            x = np.expand_dims(x, axis=-1)
            deltas, labels = self.model.apply(self.variables, x, False)
            y = np.concatenate((deltas, labels), axis=-1)
            y = np.moveaxis(y, -1, 1)

            return y

        def process(tiles):

            y = jitted(tiles)
            y = onp.asarray(y)

            return y

        process(np.zeros((self.batch_size, 256, 256)))
        self.process = transform(process)

        def postprocess(tile, scale, min_distance, threshold):

            deltas = np.moveaxis(tile[:2], 0, -1)
            labels = np.moveaxis(tile[2:3], 0, -1)
            coords, adjusted_counts = spots.compute_spot_coordinates(deltas, labels, scale=scale,
                                                                     min_distance=min_distance, threshold=threshold)

            return coords, adjusted_counts

        dummy_output = onp.zeros((3, 256, 256))
        dummy_output[2, 0, 0] = 1
        postprocess(dummy_output, 1, 1, 0.75)
        del dummy_output
        self.postprocess = transform(postprocess, output_type=('tiled_coords', 'tiled_image'))

    def predict(self, x, scale=1, min_distance=1, threshold=1.5):

        x, shape, batch_axis = self.preprocess(x, scale)

        dt = deeptile.load(x, link_data=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).compute()
        tiles = dt.process(tiles, self.process, batch_size=self.batch_size, batch_axis=batch_axis, pad_final_batch=True)
        y = dt.stitch(tiles, stitch.stitch_tiles(blend=True, sigma=5))
        y = resize(y, output_shape=(*y.shape[:-2], shape[-2], shape[-1]), order=1, preserve_range=True)
        y[..., :2, :, :] = y[..., :2, :, :] / scale

        dt2 = deeptile.load(y, link_data=False)
        tiles2 = dt2.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).compute()
        coords, adjusted_counts = dt2.process(tiles2, partial(self.postprocess,
                                              scale=scale, min_distance=min_distance, threshold=threshold),
                                              batch_axis=batch_axis)
        coords = dt2.stitch(coords, stitch.stitch_coords())
        adjusted_counts = dt2.stitch(adjusted_counts, stitch.stitch_tiles(blend=True, sigma=5))
        y = np.concatenate((y, onp.expand_dims(adjusted_counts, -3)), axis=-3)

        return coords, y


class _CelloriLoG(CelloriSpots):

    def __init__(self, model, batch_size=None):

        from cellori.applications.spots import baseline

        self.model_name = model
        self.model = None
        self.variables = None
        self.batch_size = None

        def process(tile, sigma):

            y = baseline.log_filter(tile, sigma)

            return y

        self.process = transform(process)

        def postprocess(tile, min_distance, threshold):

            coords = baseline.threshold_local_max(tile, min_distance=min_distance, threshold=threshold)

            return coords

        self.postprocess = transform(postprocess, output_type='tiled_coords')

    def predict(self, x, scale=1, sigma=1, min_distance=1, threshold=0.05):

        x, shape, batch_axis = self.preprocess(x, scale)

        dt = deeptile.load(x, link_data=False)
        tiles = dt.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).compute()
        tiles = dt.process(tiles, partial(self.process, sigma=sigma),
                           batch_size=self.batch_size, batch_axis=batch_axis, pad_final_batch=True)
        y = dt.stitch(tiles, stitch.stitch_tiles(blend=True, sigma=5))
        y = resize(y, output_shape=(*y.shape[:-2], shape[-2], shape[-1]), order=1, preserve_range=True)

        dt2 = deeptile.load(y, link_data=False)
        tiles2 = dt2.get_tiles(tile_size=(256, 256), overlap=(0.1, 0.1)).compute()
        tiles2 = dt2.process(tiles2, partial(self.postprocess,
                                             min_distance=min_distance, threshold=threshold), batch_axis=batch_axis)
        coords = dt2.stitch(tiles2, stitch.stitch_coords())

        return coords, y
