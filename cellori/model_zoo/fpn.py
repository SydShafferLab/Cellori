import jax.numpy as np

from dataclasses import field
from flax import linen as nn
from jax import image
from typing import Any, Callable

ModuleDef = Any


class FPNBlock(nn.Module):
    conv: ModuleDef = nn.Conv
    convt: ModuleDef = nn.ConvTranspose
    upsample: str = 'interpolate'
    name: str = None

    @nn.compact
    def __call__(self, x, skip):

        # Upsample pyramid level
        if self.upsample == 'interpolate':
            shape = (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3])
            x = image.resize(x, shape=shape, method='nearest')
        elif self.upsample == 'conv':
            x = self.convt(
                features=x.shape[-1],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name=self.name + 'upsample_conv'
            )(x)

        # 1x1 convolution on skip connection
        skip = self.conv(
            features=x.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            padding='SAME',
            name=self.name + 'skip_conv'
        )(skip)

        x = x + skip

        return x


class FPN(nn.Module):
    backbone: ModuleDef
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.activation.relu
    backbone_levels: list = field(default_factory=list)
    upsample: str = 'interpolate'

    @nn.compact
    def __call__(self, x, train: bool = True):

        # Get backbone outputs
        _, outputs = self.backbone()(x, train=train, capture_list=self.backbone_levels)
        backbone_levels = sorted(outputs.keys(), reverse=True)
        final_output = outputs[backbone_levels.pop(0)][0]

        # Create list to store feature maps
        features = []

        # 1x1 convolution on final backbone output
        f = self.conv(
            features=final_output.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            padding='SAME',
            name='output_conv'
        )(final_output)
        features.append(f)

        # Run FPNBlock for remaining pyramid levels
        for backbone_level in backbone_levels:
            f = FPNBlock(
                upsample=self.upsample,
                name='P{}_'.format(backbone_level[1:])
            )(f, outputs[backbone_level][0])
            features.append(f)

        bottom_shape = features[-1].shape

        # Resize feature maps
        for i in range(len(features[:-1])):
            features[i] = image.resize(features[i], shape=bottom_shape, method='nearest')

        # 3x3 convolution on each feature map
        for i in range(len(features)):
            features[i] = self.conv(
                features=bottom_shape[-1],
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                name="P{}_conv".format(len(features) - i)
            )(features[i])

        # Aggregate feature maps
        agg_features = np.sum(np.array(features), axis=0)

        # Final phase
        agg_features = self.conv(
            features=bottom_shape[-1],
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            name="final_conv"
        )(agg_features)
        agg_features = self.norm(
            use_running_average=not train,
            name="final_bn"
        )(agg_features)
        agg_features = self.act(agg_features)
        final_shape = (bottom_shape[0], 2 * bottom_shape[1], 2 * bottom_shape[2], bottom_shape[3])
        agg_features = image.resize(agg_features, shape=final_shape, method='nearest')

        return agg_features
