import jax.numpy as np

from dataclasses import field
from flax import linen as nn
from functools import partial
from jax import image
from typing import Any, Callable, Tuple

ModuleDef = Any


class FPNBlock(nn.Module):
    conv: ModuleDef = nn.Conv
    convt: ModuleDef = nn.ConvTranspose
    dense: ModuleDef = nn.Dense
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish
    upsample: str = 'interpolate'
    name: str = None

    @nn.compact
    def __call__(self, x, skip, styles):

        # Upsample pyramid level
        if self.upsample == 'interpolate':
            shape = (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3])
            x = image.resize(x, shape=shape, method='nearest')
            x = self.conv(
                features=x.shape[-1],
                kernel_size=(3, 3),
                name=self.name + 'upsample_conv'
            )(x)
        elif self.upsample == 'conv':
            x = self.convt(
                features=x.shape[-1],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name=self.name + 'upsample_convt'
            )(x)
        x = self.norm(
            name=self.name + 'upsample_bn'
        )(x)
        x = self.act(x)

        # 1x1 convolution on skip connection
        skip = self.conv(
            features=x.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            padding='SAME',
            name=self.name + 'skip_conv'
        )(skip)

        x = x + skip

        if styles is not None:
            bias = self.dense(
                features=x.shape[-1],
                name=self.name + 'styles_dense'
            )(styles)[:, None, None, :]
            x = x + bias

        return x


class FPN(nn.Module):
    backbone: ModuleDef
    conv: ModuleDef = nn.Conv
    dense: ModuleDef = nn.Dense
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish
    backbone_levels: list = field(default_factory=list)
    backbone_args: dict = field(default_factory=dict)
    add_styles: bool = False
    upsample: str = 'interpolate'
    aggregate_mode: str = 'sum'
    final_shape: Tuple[int, int] = (256, 256)

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

        # Get backbone outputs
        _, outputs = self.backbone(**self.backbone_args)(x, train=train, capture_list=self.backbone_levels)
        backbone_levels = sorted(outputs.keys(), reverse=True)
        final_output = outputs[backbone_levels.pop(0)][0]

        if self.add_styles:
            styles = np.sum(final_output, axis=(1, 2))
            styles = styles / np.linalg.norm(styles)
        else:
            styles = None

        # Create list to store feature maps
        features = []

        # 1x1 convolution on final backbone output
        f = conv(
            features=final_output.shape[-1],
            name='output_conv'
        )(final_output)
        features.append(f)

        # Run FPNBlock for remaining pyramid levels
        for backbone_level in backbone_levels:
            f = FPNBlock(
                conv=conv,
                norm=norm,
                act=self.act,
                upsample=self.upsample,
                name='P{}_'.format(backbone_level[1:])
            )(f, outputs[backbone_level][0], styles)
            features.append(f)

        bottom_shape = features[-1].shape

        # Resize feature maps
        for i in range(len(features[:-1])):
            features[i] = image.resize(features[i], shape=bottom_shape, method='nearest')

        # 3x3 convolution on each feature map
        for i in range(len(features)):
            features[i] = conv(
                features=bottom_shape[-1],
                kernel_size=(3, 3),
                name="P{}_conv".format(len(features) - i)
            )(features[i])
            features[i] = norm(
                name="P{}_bn".format(len(features) - i)
            )(features[i])
            features[i] = self.act(features[i])

            if styles is not None:
                bias = self.dense(
                    features=bottom_shape[-1],
                    name="P{}_styles_dense".format(len(features) - i)
                )(styles)[:, None, None, :]
                features[i] = features[i] + bias

        # Aggregate feature maps
        if self.aggregate_mode == 'sum':
            agg_features = np.sum(np.array(features), axis=0)
        elif self.aggregate_mode == 'concatenate':
            agg_features = np.concatenate(features, axis=-1)

        # Final phase
        agg_features = conv(
            features=bottom_shape[-1],
            kernel_size=(3, 3),
            name="final_conv"
        )(agg_features)
        agg_features = norm(
            name="final_bn"
        )(agg_features)
        agg_features = self.act(agg_features)
        if bottom_shape[1:3] != self.final_shape:
            final_shape = (bottom_shape[0], *self.final_shape, bottom_shape[3])
            agg_features = image.resize(agg_features, shape=final_shape, method='bilinear')

        return agg_features
