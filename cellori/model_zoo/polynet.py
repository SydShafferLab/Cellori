from flax import linen as nn
from functools import partial
from typing import Any, Callable, Tuple

ModuleDef = Any


class PolyHead(nn.Module):

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act_dense: Callable = nn.swish
    act_final: Callable = nn.softmax
    num_dense: int = 128
    num_classes: int = 3
    name: str = None

    @nn.compact
    def __call__(self, x):

        x = self.conv(
            features=self.num_dense,
            name=self.name + 'dense_conv'
        )(x)
        x = self.norm(
            name=self.name + 'dense_bn'
        )(x)
        x = self.act_dense(x)

        x = self.conv(
            features=self.num_classes,
            name=self.name + 'final_conv'
        )(x)
        x = self.act_final(x)

        return x


class PolyNet(nn.Module):

    fpn: ModuleDef
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    semantic_heads: Tuple[int] = (1, 3)

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

        # Process aggregate feature map through semantic heads
        for i, num_classes in enumerate(self.semantic_heads):
            f = PolyHead(
                conv=conv,
                norm=norm,
                final_act=nn.activation.softmax if num_classes > 1 else nn.activation.swish,
                num_classes=num_classes,
                name='semantic{}_'.format(i + 1)
            )(agg_features)
            poly_features.append(f)

        return poly_features
