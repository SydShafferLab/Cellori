from collections.abc import Sequence
from flax import linen as nn
from functools import partial
from typing import Any, Callable, Tuple

ModuleDef = Any


class PolyHead(nn.Module):

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act_dense: Callable = nn.swish
    act_final: Callable = nn.softmax
    num_classes: int = 3
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


class PolyNet(nn.Module):

    fpn: ModuleDef
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    semantic_heads: Tuple = ((2, None), (1, nn.sigmoid))
    return_agg: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):

        conv = partial(
            self.conv,
            kernel_size=(1, 1),
            strides=1,
            padding='SAME'
        )

        norm = partial(
            self.norm,
            use_running_average=not train
        )

        # Get aggregate feature maps from FPN
        agg_features = self.fpn()(x, train=train)

        # Create list to store feature maps
        poly_features = []

        # Process aggregate feature map through poly heads
        for i, semantic_head in enumerate(self.semantic_heads):

            name = 'poly{}_head'.format(i + 1)

            if isinstance(semantic_head, Sequence):

                num_classes, act_final = semantic_head

                f = PolyHead(
                    conv=conv,
                    norm=norm,
                    act_final=act_final,
                    num_classes=num_classes,
                    name=name
                )(agg_features)

            else:

                f = semantic_head(
                    conv=conv,
                    norm=norm,
                    name=name
                )(agg_features)

            poly_features.append(f)

        if self.return_agg:
            poly_features.append(agg_features)

        return poly_features
