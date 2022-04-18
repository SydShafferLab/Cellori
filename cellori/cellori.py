from functools import partial
from flax import linen as nn

from cellori.model_zoo import EfficientNetV2S
from cellori.model_zoo import FPN
from cellori.model_zoo import PolyNet

EfficientFPN = partial(
    FPN,
    backbone=EfficientNetV2S,
    backbone_levels={'C1', 'C2', 'C3', 'C5', 'C6'},
    upsample='interpolate'
)

Cellori = partial(
    PolyNet,
    fpn=EfficientFPN,
    semantic_heads=((1, nn.sigmoid), (3, nn.softmax))
)
