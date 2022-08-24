from flax import linen as nn
from functools import partial
from typing import Any, Callable

from cellori.model_zoo import FPN
from cellori.model_zoo import OriginNetS
from cellori.model_zoo import PolyNet

ModuleDef = Any


class RegressionHead(nn.Module):

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act_dense: Callable = nn.swish
    repeats_dense: int = 4
    num_dense: int = 256
    num_classes: int = 2
    name: str = None

    @nn.compact
    def __call__(self, x):

        for i in range(self.repeats_dense):

            x = self.conv(
                features=x.shape[-1],
                name='dense{}_conv'.format(i + 1)
            )(x)
            x = self.norm(
                name='dense{}_bn'.format(i + 1)
            )(x)
            x = self.act_dense(x)

        x = self.conv(
            features=self.num_classes,
            name='final_conv'
        )(x)

        return x


class ClassificationHead(nn.Module):

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act_dense: Callable = nn.swish
    act_final: Callable = nn.sigmoid
    num_dense: int = 128
    num_classes: int = 1
    name: str = None

    @nn.compact
    def __call__(self, x):

        x = self.conv(
            features=x.shape[-1],
            name='dense_conv'
        )(x)
        x = self.norm(
            name='dense_bn'
        )(x)
        x = self.act_dense(x)

        x = self.conv(
            features=self.num_classes,
            name='final_conv'
        )(x)
        if self.act_final is not None:
            x = self.act_final(x)

        return x


OriginFPN = partial(
    FPN,
    backbone=OriginNetS,
    backbone_levels={'C1', 'C2', 'C3', 'C4'},
    backbone_args={'stem_strides': 1},
    add_styles=True,
    ftt=False,
    upsample='interpolate',
    aggregate_mode='sum',
    final_shape=(256, 256)
)

CelloriSpotsModel = partial(
    PolyNet,
    fpn=OriginFPN,
    semantic_heads=((2, None), (1, nn.sigmoid))
)
