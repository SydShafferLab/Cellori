from flax import linen as nn
from typing import Any, Callable, Tuple

ModuleDef = Any


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
