from functools import partial
from flax import linen as nn

from cellori.model_zoo import EfficientNetV2XS
from cellori.model_zoo import FPN
from cellori.model_zoo import PolyNet

backbone = partial(
    EfficientNetV2XS,
    drop_connect_rate=0
)

EfficientFPN = partial(
    FPN,
    backbone=backbone,
    backbone_levels={'C1', 'C2', 'C3', 'C4'},
    upsample='interpolate',
    aggregate_mode='concatenate',
    final_shape=(256, 256)
)

CelloriCytoModel = partial(
    PolyNet,
    fpn=EfficientFPN,
    semantic_heads=((1, None), (1, None), (1, nn.sigmoid))
)
