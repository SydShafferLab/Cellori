import deeptile
import jax.numpy as np
import numpy as onp

from deeptile.algorithms import transform
from deeptile.extensions import stitch
from flax.training import checkpoints
from functools import partial
from jax import jit
from pathlib import Path
from skimage.transform import resize

TRAINED_MODELS_DIR = Path(__file__).parent.joinpath('trained_models')


class CelloriSegmentation:

    def __init__(self, model='cyto', batch_size=8):

        self.batch_size = batch_size

        if model == 'cyto':

            from cellori.applications.cyto.model import CelloriCytoModel
            from cellori.utils import masks

            self.model = CelloriCytoModel()
            self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath('cyto'), None)

            @jit
            def jitted(x):

                x = np.moveaxis(x, 1, -1)
                flows, semantic = self.model.apply(self.variables, x, False)
                y = np.concatenate((flows, semantic), axis=-1)

                return y

            def process(x):

                shape = x.shape
                x = resize(x, output_shape=(shape[0], 2, 256, 256), order=1, preserve_range=True)
                y = jitted(x)
                y = np.moveaxis(y, -1, 1)
                y = resize(y, output_shape=(shape[0], 3, shape[2], shape[3]), order=1, preserve_range=True)

                return y

            process(np.zeros((self.batch_size, 2, 256, 256)))
            self.process = process

            def postprocess(y, cellprob_threshold, flow_threshold):

                flows = onp.array(y[:2])
                semantic = onp.array(y[2])
                mask, _ = masks.compute_masks_dynamics(flows, semantic, cellprob_threshold=cellprob_threshold,
                                                       flow_threshold=flow_threshold)

                return mask

            self.postprocess = postprocess

    def predict(self, x, diameter=30, cellprob_threshold=0.5, flow_threshold=0.5):

        ndim = x.ndim

        if ndim == 3:
            batch_axis = None
            x = (x - onp.min(x)) / (onp.ptp(x) + 1e-7)
            x = onp.pad(x, ((0, 0), (2, 2), (2, 2)))
        elif ndim == 4:
            batch_axis = 0
            x = (x - onp.min(x, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))) / \
                (onp.ptp(x, axis=(1, 2, 3)).reshape((-1, 1, 1, 1)) + 1e-7)
            x = onp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))
        else:
            raise ValueError("Input does not have the correct dimensions.")

        if diameter != 30:
            tile_size = tuple(onp.rint(onp.array([256, 256]) * (diameter / 30)).astype(int))
        else:
            tile_size = (256, 256)

        dt = deeptile.load(x)
        dt.configure(tile_size=tile_size, overlap=(0.1, 0.1))

        tiles = dt.get_tiles()
        tiles = dt.process(tiles, transform(self.process, vectorized=True),
                           batch_size=self.batch_size, batch_axis=batch_axis, pad_final_batch=True)
        y = dt.stitch(tiles, stitch.stitch_tiles(blend=True, sigma=5))[..., 2:-2, 2:-2]

        if ndim == 3:
            mask = self.postprocess(y, cellprob_threshold, flow_threshold)
        else:
            mask_list = []
            for i in range(len(y)):
                mask_list.append(self.postprocess(y[i], cellprob_threshold, flow_threshold))
            mask = np.stack(mask_list)

        return mask, y


class CelloriSpots:

    def __init__(self, model='spots', batch_size=8):

        self.batch_size = batch_size

        if model == 'spots':

            from cellori.applications.spots.model import CelloriSpotsModel
            from cellori.utils import spots

            self.model = CelloriSpotsModel()
            self.variables = checkpoints.restore_checkpoint(TRAINED_MODELS_DIR.joinpath('spots'), None)

            @jit
            def jitted(x):

                x = np.moveaxis(x, 1, -1)
                deltas, labels = self.model.apply(self.variables, x, False)
                y = np.concatenate((deltas, labels), axis=-1)

                return y

            def process(x):

                shape = x.shape
                x = resize(x, output_shape=(shape[0], 1, 256, 256), order=1, preserve_range=True)
                y = jitted(x)
                y = np.moveaxis(y, -1, 1)
                y = resize(y, output_shape=(shape[0], 3, shape[2], shape[3]), order=1, preserve_range=True)

                return y

            process(np.zeros((self.batch_size, 1, 256, 256)))
            self.process = process

            def postprocess(y, min_distance, threshold):

                deltas = np.moveaxis(y[:2], 0, -1)
                labels = np.moveaxis(y[2:3], 0, -1)
                coords = spots.compute_spot_coordinates(deltas, labels, min_distance=min_distance, threshold=threshold)

                return coords

            dummy_output = onp.zeros((3, 256, 256))
            dummy_output[2, 0, 0] = 1
            postprocess(dummy_output, 1, 0.75)
            del dummy_output
            self.postprocess = postprocess

    def predict(self, x, scale=1, min_distance=1, threshold=0.75):

        ndim = x.ndim

        if ndim == 2:
            batch_axis = None
            x = (x - onp.min(x)) / (onp.ptp(x) + 1e-7)
        elif ndim == 3:
            batch_axis = 0
            x = (x - onp.min(x, axis=(1, 2)).reshape((-1, 1, 1))) / \
                (onp.ptp(x, axis=(1, 2)).reshape((-1, 1, 1)) + 1e-7)
        else:
            raise ValueError("Input does not have the correct dimensions.")

        x = onp.expand_dims(x, axis=-3)

        if scale != 1:
            tile_size = tuple(onp.rint(onp.array([256, 256]) / scale).astype(int))
        else:
            tile_size = (256, 256)

        dt = deeptile.load(x)
        dt.configure(tile_size=tile_size, overlap=(0.1, 0.1))

        tiles = dt.get_tiles()
        tiles = dt.process(tiles, transform(self.process, vectorized=True),
                           batch_size=self.batch_size, batch_axis=batch_axis, pad_final_batch=True)
        y = dt.stitch(tiles, stitch.stitch_tiles(blend=True, sigma=5))

        dt2 = deeptile.load(y)
        dt2.configure(tile_size=(256, 256), overlap=(0.1, 0.1))

        tiles2 = dt2.get_tiles()
        tiles2 = dt2.process(tiles2, transform(partial(self.postprocess,
                                                       min_distance=min_distance, threshold=threshold),
                                               vectorized=False, output_type='tiled_coords'), batch_axis=batch_axis)
        coords = dt.stitch(tiles2, stitch.stitch_coords())

        if ndim == 2:
            coords = coords[0]
        else:
            coords = onp.asarray(coords)

        return coords, y
