import jax.numpy as np
from einops import rearrange
from functools import partial
from flax import linen as nn
from jax import image
from typing import Any, Callable, Tuple

ModuleDef = Any


class UpConvBlock(nn.Module):

    conv: ModuleDef = nn.Conv
    convt: ModuleDef = nn.ConvTranspose
    features: int = None
    upsample: str = 'interpolate'
    name: str = None

    @nn.compact
    def __call__(self, x):

        if self.features is None:
            features = x.shape[-1]
        else:
            features = self.features

        if self.upsample == 'interpolate':
            shape = (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3])
            x = image.resize(x, shape=shape, method='nearest')
            x = self.conv(
                features=features,
                kernel_size=(3, 3),
                name='upsample_conv'
            )(x)
        elif self.upsample == 'conv':
            x = self.convt(
                features=features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name='upsample_convt'
            )(x)

        return x


class PixelShuffle(nn.Module):
    scale_factor: int

    def setup(self):
        self.layer = partial(
            rearrange,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.layer(x)


class MBConvBlock(nn.Module):
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = nn.swish
    input_filters: int = 32
    output_filters: int = 16
    expand_ratio: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    se_ratio: float = 0.0
    dropout_rate: float = 0.2
    deterministic: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        residual = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = self.conv(
                features=filters,
                name="expand_conv",
            )(x)
            x = self.norm(
                name="expand_bn"
            )(x)
            x = self.act(x)

        # Depthwise conv
        x = self.conv(
            features=x.shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides,
            feature_group_count=x.shape[-1],
            name='dwconv'
        )(x)
        x = self.norm(
            name="bn"
        )(x)
        x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = nn.avg_pool(x, x.shape[1:3])

            se = self.conv(
                features=filters_se,
                padding='SAME',
                use_bias=True,
                name='se_reduce'
            )(se)
            se = self.act(se)
            se = self.conv(
                features=filters,
                use_bias=True,
                name='se_expand'
            )(se)
            se = nn.sigmoid(se)

            x = x * se

        # Output phase
        x = self.conv(
            features=self.output_filters,
            name='project_conv'
        )(x)
        x = self.norm(
            name="project_bn"
        )(x)

        # Residual
        if (self.strides == 1) and (self.input_filters == self.output_filters):
            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    name='drop'
                )(x)
            x = x + residual

        return x


class FusedMBConvBlock(nn.Module):
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = nn.swish
    input_filters: int = 32
    output_filters: int = 16
    expand_ratio: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    se_ratio: float = 0.0
    dropout_rate: float = 0.2
    deterministic: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        residual = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = self.conv(
                features=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                name="expand_conv",
            )(x)
            x = self.norm(
                name="expand_bn"
            )(x)
            x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = nn.avg_pool(x, x.shape[1:3])

            se = self.conv(
                features=filters_se,
                use_bias=True,
                name='se_reduce'
            )(se)
            se = self.act(se)
            se = self.conv(
                features=filters,
                use_bias=True,
                name='se_expand'
            )(se)
            se = nn.sigmoid(se)

            x = x * se

        # Output phase
        x = self.conv(
            features=self.output_filters,
            kernel_size=(1, 1) if self.expand_ratio != 1 else self.kernel_size,
            strides=1 if self.expand_ratio != 1 else self.strides,
            name='project_conv'
        )(x)
        x = self.norm(
            name="project_bn"
        )(x)
        if self.expand_ratio == 1:
            x = self.act(x)

        # Residual
        if (self.strides == 1) and (self.input_filters == self.output_filters):
            if self.dropout_rate > 0:
                x = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    name='drop'
                )(x)
            x = x + residual

        return x


class FeatureTextureTransfer(nn.Module):
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = nn.swish
    repeats: int = 4
    expand_ratio: int = 4
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    se_ratio: float = 0.25
    dropout_rate: float = 0.2
    deterministic: bool = False
    name: str = None

    @nn.compact
    def __call__(self, p2, p3):

        # Content Extractor
        input_filters = p3.shape[-1]
        output_filters = p3.shape[-1] * 4
        for i in range(self.repeats):
            p3 = MBConvBlock(
                conv=self.conv,
                norm=self.norm,
                act=self.act,
                input_filters=input_filters,
                output_filters=output_filters,
                expand_ratio=self.expand_ratio,
                kernel_size=self.kernel_size,
                strides=self.strides,
                se_ratio=self.se_ratio,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                name='content_extractor{}'.format(i + 1)
            )(p3)

        # Subpixel convolution
        p3 = PixelShuffle(
            scale_factor=2
        )(p3)

        p3p = np.concatenate((p2, p3), axis=-1)

        # Texture Extractor
        input_filters = p3p.shape[-1]
        output_filters = round(p3p.shape[-1] / 2)
        for i in range(self.repeats):
            p3p = MBConvBlock(
                conv=self.conv,
                norm=self.norm,
                act=self.act,
                input_filters=input_filters,
                output_filters=output_filters,
                expand_ratio=self.expand_ratio,
                kernel_size=self.kernel_size,
                strides=self.strides,
                se_ratio=self.se_ratio,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                name='texture_extractor{}'.format(i + 1)
            )(p3p)

        p3p = p3p + p3

        return p3p
