import deeptile
import jax.numpy as np
import numpy as onp

from deeptile.example_algorithms import stitch
from functools import partial
from flax.training import checkpoints
from jax import jit

from cellori.utils import masks


class Cellori:

    def __init__(self, model='cyto'):

        if model == 'cyto':

            from cellori.applications.cyto.model import CelloriCytoModel

            self.model = CelloriCytoModel()
            self.variables = checkpoints.restore_checkpoint('cellori_model', None)

            @partial(jit, static_argnums=1)
            def apply(x, batch_size=8):

                if x.shape[0] == batch_size:

                    x = np.moveaxis(x, 1, -1)
                    x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)))
                    flows, semantic = self.model.apply(self.variables, x, False)
                    y = np.concatenate((flows, semantic), axis=-1)
                    y = y[:, 2:-2, 2:-2]

                    return y

                else:

                    assert ValueError('Batch size mismatch.')

            apply(np.zeros((1, 2, 252, 252)), 1)
            self.apply = apply

            def postprocess(x, cellprob_threshold, flow_threshold):

                flows = onp.array(x[:, :, :2])
                semantic = onp.array(x[:, :, 2])
                flows = onp.moveaxis(flows, -1, 0)
                mask, _ = masks.compute_masks_dynamics(flows, semantic, cellprob_threshold=cellprob_threshold,
                                                       flow_threshold=flow_threshold)

                return mask

            self.postprocess = postprocess

    def predict(self, x, cellprob_threshold=0.5, flow_threshold=0.5):

        dt = deeptile.load(x)
        dt.configure(tile_size=(252, 252), overlap=(0.25, 0.25))

        tiles = dt.get_tiles()
        tiles = dt.process(tiles, self.apply)
        tiles = dt.process(tiles, partial(self.postprocess, cellprob_threshold=cellprob_threshold,
                                          flow_threshold=flow_threshold))

        mask = dt.stitch(tiles, stitch.stitch_masks())

        return mask
