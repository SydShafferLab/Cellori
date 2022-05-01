from functools import partial
from flax import linen as nn

from cellori.model_zoo import FPN
from cellori.model_zoo import OriginNetM
from cellori.model_zoo import PolyNet

OriginFPN = partial(
    FPN,
    backbone=OriginNetM,
    backbone_levels={'C1', 'C2', 'C3', 'C4', 'C5'},
    backbone_args={'stem_strides': 1},
    add_styles=True,
    upsample='interpolate',
    aggregate_mode='sum',
    final_shape=(256, 256)
)

CelloriCytoModel = partial(
    PolyNet,
    fpn=OriginFPN,
    semantic_heads=((2, None), (1, nn.sigmoid))
)
