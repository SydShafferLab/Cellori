import deeptile
import jax.numpy as np
import numpy as onp

from deeptile.algorithms import transform
from deeptile.example_algorithms import stitch
from functools import partial
from flax.training import checkpoints
from jax import jit
from skimage.transform import resize

from cellori.utils import masks


class Cellori:

    def __init__(self, model='cyto', batch_size=8):

        self.batch_size = batch_size

        if model == 'cyto':

            from cellori.applications.cyto.model import CelloriCytoModel

            self.model = CelloriCytoModel()
            self.variables = checkpoints.restore_checkpoint('cellori_model', None)

            @jit
            def jitted(x):

                x = np.moveaxis(x, 1, -1)
                x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)))
                flows, semantic = self.model.apply(self.variables, x, False)
                y = np.concatenate((flows, semantic), axis=-1)
                y = y[:, 2:-2, 2:-2]

                return y

            def process(x):

                x = resize(x, output_shape=(x.shape[0], 2, 252, 252), order=1, preserve_range=True)
                y = jitted(x)

                return y

            process(np.zeros((self.batch_size, 2, 252, 252)))
            self.process = process

            def postprocess(x, tile_size, cellprob_threshold, flow_threshold):

                flows = onp.array(x[:, :, :2])
                flows = onp.moveaxis(flows, -1, 0)
                flows = resize(flows, output_shape=(2, *tile_size), order=1, preserve_range=True)
                semantic = onp.array(x[:, :, 2])
                semantic = resize(semantic, output_shape=tile_size, order=1, preserve_range=True)
                mask, _ = masks.compute_masks_dynamics(flows, semantic, cellprob_threshold=cellprob_threshold,
                                                       flow_threshold=flow_threshold)

                return mask

            self.postprocess = postprocess

    def predict(self, x, diameter=30, cellprob_threshold=0.5, flow_threshold=0.5):

        if diameter != 30:
            tile_size = tuple(onp.rint(onp.array([252, 252]) * (diameter / 30)).astype(int))
        else:
            tile_size = (252, 252)

        dt = deeptile.load(x)
        dt.configure(tile_size=tile_size, overlap=(0.25, 0.25))

        tiles = dt.get_tiles()
        tiles = dt.process(tiles, transform(self.process, batch=True), batch_size=self.batch_size)
        tiles = dt.process(tiles, transform(partial(self.postprocess, tile_size=tile_size,
                                                    cellprob_threshold=cellprob_threshold,
                                                    flow_threshold=flow_threshold), batch=False),
                           batch_size=self.batch_size)

        mask = dt.stitch(tiles, stitch.stitch_masks())

        return mask
